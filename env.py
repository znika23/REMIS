import asyncio
import os.path
import copy
import numpy as np
import tkinter as tk
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from server import Edge_Server
from utils import MessageBroker
from __init__ import params_data_dir, map_data_dir
from utils import time2int, ReplayBuffer, index2action


class Env:
    finished_task = 0
    success_task = 0
    action = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    reward_record = np.zeros((10, 2, 2))

    def __init__(self, params, QNet) -> None:
        #
        self.params_alg, self.params_map, self.params_edge = params['alg'], params['map'], params['edge']
        self.rng = np.random.default_rng(self.params_alg['seed'])
        self.broker = MessageBroker()
        self.buffer = ReplayBuffer(params['alg'])
        self.QNet = QNet
        # config
        (
            self.server_coordinates, self.server_reliability, self.server_AI_power,
            self.experts_coordinates, self.experts_utilization,
            self.edge_servers
        ) = self.config()
        print('expert_coordinates=\n', self.experts_coordinates)
        print('server_reliability=\n', self.server_reliability)

    def get_init_state(self, gating_result, result):
        time = time2int()
        layer = 0
        # utilization = self.experts_utilization[0][gating_result]
        utilization = copy.deepcopy(self.experts_utilization[0][gating_result])
        reliability = np.zeros(self.params_edge['relax-k'], dtype=float)
        for i in range(self.params_edge['relax-k']):
            utilization[i] = max((utilization[i] + 5) / 6, 0)
            server = self.experts_coordinates[0][gating_result[i]]
            reliability[i] = self.server_reliability[0][server] * 5
        state = np.concatenate(([time, layer, result], utilization, reliability))
        return state

    def config(self):
        """
        environment setup
        """
        # params preparation
        layers, experts, servers = self.params_map['layer'], self.params_map['expert'], self.params_map[
            'server_per_layer']
        server_range = self.params_map['map_size'] / servers
        server_size = server_range / 5

        # randomly set servers' coordinates , reliability , AI_power and experts' distribution
        server_coordinates, server_reliability, server_AI_power = (
            np.zeros((layers, servers, 2)), np.zeros((layers, servers)), np.zeros((layers, servers))
        )
        experts_coordinates, expert_utilization = np.zeros((layers, experts), dtype=int), np.ones((layers, experts))
        for layer in range(layers):
            expert_ids = np.arange(0, experts)
            for server in range(servers):
                # servers' coordinates
                x = self.rng.uniform(layer * server_range + server_size, (layer + 1) * server_range - server_size)
                y = self.rng.uniform(server * server_range + server_size, (server + 1) * server_range - + server_size)
                # servers' reliability
                reliability = self.rng.choice(self.params_edge['reliability'])
                # servers' AI_power
                AI_power = self.rng.choice(self.params_edge['AI_power'])
                # experts' distribution
                selected_expert = self.rng.choice(expert_ids, size=experts // servers, replace=False)
                expert_ids = np.setdiff1d(expert_ids, selected_expert)
                #
                server_coordinates[layer, server] = (x, y)
                server_reliability[layer, server] = reliability
                server_AI_power[layer, server] = AI_power
                experts_coordinates[layer, selected_expert] = server

        # construct servers' object
        edge_servers = np.empty((layers, servers), dtype=Edge_Server)
        for layer in range(layers):
            for server in range(servers):
                # edge servers' parameters
                params = copy.deepcopy(self.params_edge)
                params['id'] = f'{layer}_{server}'
                params['server_coordinates'] = {
                    'same_layer': server_coordinates[layer],
                    'next_layer': server_coordinates[layer + 1] if layer != layers - 1 else None
                }
                params['server_reliability'] = {
                    'same_layer': server_reliability[layer],
                    'next_layer': server_reliability[layer + 1] if layer != layers - 1 else None
                }
                params['server_AI_power'] = {
                    'same_layer': server_AI_power[layer],
                    'next_layer': server_AI_power[layer + 1] if layer != layers - 1 else None
                }
                params['expert_coordinates'] = {
                    'same_layer': experts_coordinates[layer],
                    'next_layer': experts_coordinates[layer + 1] if layer != layers - 1 else None
                }
                params['expert_utilization'] = {
                    'same_layer': expert_utilization[layer],
                    'next_layer': expert_utilization[layer + 1] if layer != layers - 1 else None
                }
                edge_servers[layer, server] = Edge_Server(params, self.broker, self.rng, self.buffer, self.QNet)
        return (
            server_coordinates, server_reliability, server_AI_power,
            experts_coordinates, expert_utilization,
            edge_servers
        )

    async def manipulate(self):
        layers, experts, select_experts, servers = (
            self.params_map['layer'], self.params_map['expert'],
            self.params_edge['relax-k'], self.params_map['server_per_layer']
        )

        for layer in range(layers):
            for server in range(servers):
                self.edge_servers[layer][server].update()
                asyncio.create_task(self.edge_servers[layer][server].handle())

        start_time = time.perf_counter()
        # create tasks
        for task in range(self.params_alg['task_batch']):
            gating_result = self.rng.choice(experts, select_experts, replace=False)
            s0 = self.get_init_state(gating_result, 0)
            a0 = self.QNet(torch.tensor(s0[1:], dtype=torch.float)).argmax().item()
            self.buffer.cache(task, s0, a0)

            index1, index2 = self.action[a0]
            select_expert_1, select_expert_2 = gating_result[index1], gating_result[index2]
            receiver_1 = f'0_{self.experts_coordinates[0][select_expert_1]}'
            receiver_2 = f'0_{self.experts_coordinates[0][select_expert_2]}'
            print("-------------------------------------------------------------")
            print('+++ task:', task, ' send to ', receiver_1, ' and', receiver_2, ' by env with expect expert =',
                  select_expert_1, " ", select_expert_2)
            await self.broker.send('-1_0', receiver_1, task, select_expert_1, select_expert_2, 0, time2int())
            if receiver_1 == receiver_2:
                continue
            await asyncio.sleep(0.1)
            await self.broker.send('-1_0', receiver_2, task, select_expert_2, select_expert_1, 0, time2int())

        while Env.finished_task != self.params_alg['task_batch']:
            await asyncio.sleep(0.1)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        print('success_task:', Env.success_task,
              'success_rate:', Env.success_task / self.params_alg['task_batch'] * 100, '%'
              'total_time:', elapsed_time,
              'avg_task_time', elapsed_time / self.params_alg['task_batch'] )
        print(f"Elapsed time: {elapsed_time * 1000:.3f} ms")
        Env.finished_task = 0
        Env.success_task = 0
        return
        # await asyncio.gather(*handling_loop)

    def view(self):
        """
        plot llm distribution map
        """
        # params preparation
        layers, experts, servers = self.params_map['layer'], self.params_map['expert'], self.params_map[
            'server_per_layer']
        fig_size = self.params_map['map_size'] / 10
        fig_wth, fig_hth = fig_size * (layers / servers), fig_size
        img_size = (fig_size / servers) / 5
        colors = ['#99C9F2', '#A9D18E', '#FFD966', '#F4B183']

        # construct the Tkinter window
        root_window = tk.Tk()
        root_window.title("CEC-MoE")

        # construct matplotlib
        fig, ax = plt.subplots(figsize=(fig_wth, fig_hth))
        ax.axis('off')
        ax.set(xlim=(0, fig_wth), ylim=(0, fig_hth))
        fig.patch.set_facecolor('#F2F2F2')  # 设置图形背景色

        # construct canvas
        canvas = FigureCanvasTkAgg(fig, master=root_window)
        widget_map = canvas.get_tk_widget()
        widget_map.grid(row=0, column=0)
        for layer in range(layers):
            for server in range(servers):
                x, y = self.server_coordinates[layer, server] / 10
                # server_icon
                server_img = mpimg.imread(os.path.join(map_data_dir, f'server_{(layer + 1) % 4}.png'))
                ax.imshow(server_img, extent=(x - img_size, x + img_size, y - img_size, y + img_size))
        root_window.mainloop()
