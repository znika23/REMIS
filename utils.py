import asyncio
import time
import numpy as np
import random
from collections import deque


def time2int():
    return time.time_ns()


def index2action(k, a, b):
    a, b = sorted([a, b])
    action = a * (k - 1) - (a * (a + 1) // 2) + (b - 1)
    return action


class ReplayBuffer:
    """
    Replay Buffer for store the samples for DDQN Training
    """

    def __init__(self, parameters) -> None:
        # self.rng = np.random.default_rng(parameters['seed'])
        self.rng = np.random.default_rng(132)
        self.total_buffer_size = parameters['buffer_size']
        # self.task_state = np.zeros((parameters['task_batch'], parameters['relax-k'] * 1 + 2))
        # self.task_state = np.zeros((parameters['task_batch'], parameters['relax-k'] * 2 + 2))
        self.task_state = np.zeros((parameters['task_batch'], parameters['relax-k'] * 2 + 3))
        self.task_action = np.zeros(parameters['task_batch'])
        self.data = []

    def __len__(self):
        return len(self.data)

    def get(self, task_id):
        return self.task_state[task_id], self.task_action[task_id]

    def cache(self, task_id, state, a):
        self.task_state[task_id] = state
        self.task_action[task_id] = a

    def add(self, sample):
        """
        Add a sample to the buffer.
        """
        sample = np.concatenate([sample[0], [sample[1]], [sample[2]], sample[3]])
        if len(self.data) == 0:
            self.data = sample[None, :]
        elif self.data.shape[0] < self.total_buffer_size:
            self.data = np.vstack((self.data, sample[None, :]))
        else:
            self.data = np.vstack((self.data[1:, :], sample[None, :]))
        return

    def sample(self, batch_size):
        """
        Randomly sample batch_size data from buffer
        """
        if len(self.data) < batch_size:
            raise Exception('Not enough buffer data for sampling.')
        else:
            # idx = self.rng.choice(self.data.shape[0], batch_size, replace=False)
            idx = random.sample(range(self.data.shape[0]), batch_size)
            return self.data[idx, :]

    def save_to_npy(self, file_path):
        np.save(file_path, self.data)

    def load_from_npy(self, file_path):
        data = np.load(file_path)
        self.data = data


class MessageBroker:
    """
    Msg Broker for multi-communication between servers
    """

    def __init__(self):
        self.queues = {}

    def register(self, node_id):
        self.queues[node_id] = asyncio.Queue()

    async def send(self, sender, receiver, task_id, expert_1, expert_2, result, time):
        if receiver in self.queues:
            await self.queues[receiver].put((sender, task_id, expert_1, expert_2, result, time))

    async def receive(self, node_id):
        if node_id in self.queues:
            return await self.queues[node_id].get()
