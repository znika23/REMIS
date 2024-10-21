import numpy as np
import os
import json
import asyncio

import torch.nn

from env import Env
from model import QNet
from __init__ import params_data_dir, samples_data_dir, models_data_dir


# noinspection NonAsciiCharacters
class DDQN_Learn:
    def __init__(self, env_type, qnet) -> None:
        params_file_name = os.path.join(params_data_dir, f'params_{env_type}.json')
        self.params = json.load(open(params_file_name))
        # model config
        relax_k = self.params['alg']['relax-k']
        # self.dims, self.dima = relax_k * 1 + 1, relax_k * (relax_k - 1) // 2
        self.dims, self.dima = relax_k * 2 + 2, relax_k * (relax_k - 1) // 2
        # self.dims, self.dima = relax_k * 2 + 1, relax_k * (relax_k - 1) // 2
        self.onlineQ = QNet(self.params['alg'], self.dims, self.dima)
        if qnet is None:
            self.targetQ = QNet(self.params['alg'], self.dims, self.dima)
        else:
            self.targetQ = qnet
        self.env = Env(self.params, self.targetQ)
        # hy_parameters
        self.device = self.params['alg']['device']
        self.σ = self.params['alg']['learning_rate']
        self.γ = self.params['alg']['reward_decay']
        self.episode = self.params['alg']['episode']
        self.batch = self.params['alg']['batch']

    def generate_action(self, s):
        """
        compute actions from online Q-Networks
        """
        s = s[None, :].float().to(self.device) if torch.is_tensor(s) else torch.from_numpy(s[None, :]).float().to(
            self.device)
        with torch.no_grad():
            Q = self.onlineQ(s)
        a = Q.argmax().item()
        return a

    async def manipulate(self):
        for i in range(50):
            await self.env.manipulate()
        samples_file_name = os.path.join(samples_data_dir, 'samples_2')
        self.env.buffer.save_to_npy(samples_file_name)

    async def train(self):
        samples_file_name = os.path.join(samples_data_dir, 'samples_2.npy')
        self.env.buffer.load_from_npy(samples_file_name)

        loss_fn = torch.nn.MSELoss()
        opt = torch.optim.Adam(self.onlineQ.parameters(), lr=self.σ)

        train_loss = np.zeros((self.episode, self.batch))
        train_rewards = np.zeros((self.episode, self.batch))

        for ep in range(self.episode):
            samples = self.env.buffer.sample(batch_size=self.batch)
            samples = torch.from_numpy(samples).float().to(self.device)
            s = samples[:, :self.dims]
            a = samples[:, self.dims].type(torch.int64).unsqueeze(1)
            r = samples[:, self.dims + 1].unsqueeze(1)
            s_new = samples[:, -self.dims:]
            done = (s_new[:, 0] == 1.0).int().unsqueeze(1)

            # compute bellman operator using target Q at snew
            with torch.no_grad():
                Q = self.onlineQ(s_new)
            idx = Q.argmax(dim=1).unsqueeze(-1)
            with torch.no_grad():
                q = self.targetQ(s_new).gather(1, idx)
            y = r + self.γ * (1 - done) * q
            loss = loss_fn(self.onlineQ(s).gather(1, a), y)

            # update online q network for one step GD
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss[ep] = loss.item()
            # print('episode=', ep, 'loss=', loss.item())
            if ep % 50 == 0:
                self.update_target_Q_parameter(1)
            if ep % 1000 == 0 and ep != 0:
                times = ep // 1000
                print(np.average(train_loss[(times - 1) * 1000:times * 1000]))
        print(np.average(train_loss[-1000:]))
        for i in range(20):
            await self.env.manipulate()
        torch.save(self.onlineQ.state_dict(), os.path.join(models_data_dir, 'time_acc_online.pt'))
        torch.save(self.onlineQ.state_dict(), os.path.join(models_data_dir, 'time_acc_target.pt'))

    def update_target_Q_parameter(self, tau):
        """
        This function replaces the target Q network parameters
        """
        d_online = self.onlineQ.state_dict()
        d_target = self.targetQ.state_dict()
        for key in d_online.keys():
            d_target[key] = (1 - tau) * d_target[key] + tau * d_online[key]
        self.targetQ.load_state_dict(d_target)
        return

DDQN_object = DDQN_Learn(2, None)
asyncio.run(DDQN_object.manipulate())




