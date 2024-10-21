import copy
import math
import os
import time
import torch
import asyncio
from collections import defaultdict
import numpy as np
from datetime import datetime

from utils import time2int, index2action


class Edge_Server:
    def __init__(self, params, broker, rng, buffer, QNet) -> None:
        # parameter preparation
        self.rng = rng
        self.broker = broker
        self.buffer = buffer
        self.QNet = QNet
        self.id = params['id']
        self.layer, self.server = map(int, params['id'].split('_'))
        self.params = params
        self.comp_time, self.trans_time = self.config()

        # expert management
        self.expert_index = np.where(params['expert_coordinates']['same_layer'] == self.id)
        self.expert_utilization = params['expert_utilization']['same_layer']
        self.lock = asyncio.Semaphore()

        # task records
        self.tasks_time = defaultdict(lambda: {'loc': 0, 'neighbor': 0})
        self.tasks_result = defaultdict(lambda: {'loc': 0, 'neighbor': 0})

    def update(self):
        self.tasks_time = defaultdict(lambda: {'loc': 0, 'neighbor': 0})
        self.tasks_result = defaultdict(lambda: {'loc': 0, 'neighbor': 0})

    # noinspection NonAsciiCharacters
    def config(self):
        """
        Register the msg broker
        Manipulate the inference time for one expert and the transmission time between servers
        unit: ms
        """
        # register msg broker
        self.broker.register(self.id)

        # comp_time
        AI_power = self.params['server_AI_power']['same_layer'][self.server]
        #  "AI_power": [1880000, 2660000, 3220000],
        η = self.params['activation']
        m, m_h = self.params['ffn_dim'], self.params['hid_dim']
        comp_time = (4 * m * m_h + 2 * m_h *m + η * m_h + m_h) / AI_power / 1e3 * 20

        # trans_time
        B, ε = self.params['bandwidth'], self.params['epsilon']
        trans_time = (m * ε) / (B * 1e6 * np.log2(1 + 10)) * 20
        return comp_time, trans_time

    def gating(self):
        """
        Manipulate the gating function
        """
        total_expert, select_experts = self.params['experts'], self.params['relax-k']
        gating_result = self.rng.choice(total_expert, select_experts, replace=False)
        return gating_result

    async def handle(self):
        while True:
            sender, task_id, expert_1, expert_2, result, time = await self.broker.receive(self.id)
            sender_layer, sender_server = map(int, sender.split('_'))

            # msg from server in the same layer / lower layer
            if self.layer == sender_layer:
                # subtask for neighbour server has been done
                self.tasks_time[task_id]['neighbor'] = time
                self.tasks_result[task_id]['neighbor'] = result
                await self.check(task_id, expert_1, expert_2)
            else:
                if self.params['expert_coordinates']['same_layer'][expert_1] == self.server:
                    asyncio.create_task(self.process_req(task_id, expert_1, expert_2, result))
                else:
                    asyncio.create_task(self.process_req(task_id, expert_2, expert_1, result))

    async def process_req(self, task_id, loc_expert, neighbor_expert, result):
        """
        manipulate the inference of each expert
        """

        reliability = self.params['server_reliability']['same_layer'][self.server]

        if self.params['expert_coordinates']['same_layer'][neighbor_expert] != self.server:
            async with self.lock:
                self.expert_utilization[loc_expert] -= 1
            await asyncio.sleep((1 - self.expert_utilization[loc_expert]) * self.comp_time)
            if self.rng.uniform(0, 1) < reliability:
                result = 1
            async with self.lock:
                self.expert_utilization[loc_expert] += 1
            self.tasks_time[task_id]['loc'] = time2int()
            self.tasks_result[task_id]['loc'] = result

        # if 2 activated expert locate in same server
        else:
            async with self.lock:
                self.expert_utilization[loc_expert] -= 1
                self.expert_utilization[neighbor_expert] -= 1
            await asyncio.sleep((1 - min(self.expert_utilization[loc_expert],
                                         self.expert_utilization[neighbor_expert])) * self.comp_time)
            if self.rng.uniform(0, 1) < reliability:
                result = 1
            async with self.lock:
                self.expert_utilization[loc_expert] += 1
                self.expert_utilization[neighbor_expert] += 1
            self.tasks_time[task_id]['loc'] = self.tasks_time[task_id]['neighbor'] = time2int()
            self.tasks_result[task_id]['loc'] = self.tasks_time[task_id]['neighbor'] = result

        await self.check(task_id, loc_expert, neighbor_expert)

    async def check(self, task_id, loc_expert, neighbor_expert):
        """
        Checks and determines whether to pass the task to the next level
        """

        result = 1 if any(value == 1 for value in self.tasks_result[task_id].values()) else 0

        # both local tasks and neighbour tasks have finished
        if all(value != 0 for value in self.tasks_time[task_id].values()):
            # local task later
            if self.tasks_time[task_id]['loc'] >= self.tasks_time[task_id]['neighbor']:
                expert_coordinates = self.params['expert_coordinates']['next_layer']

                if expert_coordinates is None:
                    s, a = self.buffer.get(task_id)
                    s_ = self.get_cur_state(np.array([0, 0, 0, 0]), result)
                    r = -(s_[0] - s[0]) * 1e-6 + (
                                self.params['res_reward'] * (1 - result) - self.params['res_reward'] * result)
                    if s_[2] - s[2] == 1:
                        r -= self.params['res_reward']
                    r = max((r + 10000) / 10200, 0)
                    self.buffer.add((s[1:], a, r, s_[1:]))
                    print("-------------------------------------------------------------")
                    print('*** task:', task_id, ' finished with result', result, 'at time', time2int(), 'at', self.id)
                    from env import Env
                    Env.finished_task += 1
                    if result == 0:
                        Env.success_task += 1
                    return

                gating_result = self.gating()
                s, a = self.buffer.get(task_id)
                s_ = self.get_cur_state(gating_result, result)
                r = -(s_[0] - s[0]) * 1e-6 + (self.params['res_reward'] * (1 - result) - self.params['res_reward'] * result)
                if s_[2] - s[2] == 1:
                    r -= self.params['res_reward']

                r = max((r + 10000) / 10200, 0)
                self.buffer.add((s[1:], a, r, s_[1:]))
                a_ = self.QNet(torch.tensor(s_[1:], dtype=torch.float)).argmax().item()
                from env import Env
                index1, index2 = Env.action[a_]
                select_expert_1, select_expert_2 = gating_result[index1], gating_result[index2]
                self.buffer.cache(task_id, s_, a_)

                receiver_1 = f'{self.layer + 1}_{expert_coordinates[select_expert_1]}'
                receiver_2 = f'{self.layer + 1}_{expert_coordinates[select_expert_2]}'
                await asyncio.sleep(self.trans_time)

                print("-------------------------------------------------------------")
                print('+++ task:', task_id, ' send to ', receiver_1, ' and', receiver_2, ' by', self.id, 'for expert',
                      select_expert_1, ' and ', select_expert_2, ' with result ', result)
                await self.broker.send(self.id, receiver_1, task_id, select_expert_1, select_expert_2, result, time2int())
                if receiver_1 == receiver_2:
                    return
                await asyncio.sleep(0.1)
                await self.broker.send(self.id, receiver_2, task_id, select_expert_2, select_expert_1, result, time2int())

        # local tasks has finished
        elif self.tasks_time[task_id]['neighbor'] == 0:
            expert_coordinates = self.params['expert_coordinates']['same_layer']
            await asyncio.sleep(self.trans_time)
            receiver = f'{self.layer}_{expert_coordinates[neighbor_expert]}'
            await self.broker.send(self.id, receiver, task_id, neighbor_expert, loc_expert, result, self.tasks_time[task_id]['loc'])

    def get_cur_state(self, gating_result, result):
        time = time2int()
        layer = (self.layer + 1) / 10
        if layer == 1:
            utilization, reliability = np.zeros(self.params['relax-k']), np.zeros(self.params['relax-k'])
        else:
            utilization = copy.deepcopy(self.params['expert_utilization']['next_layer'][gating_result])
            reliability = np.zeros(self.params['relax-k'], dtype=float)
            for i in range(self.params['relax-k']):
                utilization[i] = max((utilization[i] + 5) / 6, 0)
                server = self.params['expert_coordinates']['next_layer'][gating_result[i]]
                reliability[i] = self.params['server_reliability']['next_layer'][server] * 5
        state = np.concatenate(([time, layer, result], utilization, reliability))
        return state

