from torch import nn
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import os

os.environ["OMP_NUM_THREADS"] = "1"


class SharedAdamOpt(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.9), eps=1e-8, weight_decay=0):

        super(SharedAdamOpt, self).__init__(params, lr=lr, betas=betas, \
                                            eps=eps, weight_decay=weight_decay)

        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(param.data)
                state['exp_avg_sq'] = torch.zeros_like(param.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class Worker(mp.Process):

    def __init__(self, network, global_net, n_s, n_a, opt, gamma, global_episode, global_episode_reward, res_queue, \
                 name, max_episode, update_global_iter, env_name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.global_episode, self.global_episode_reward, self.res_queue = global_episode, global_episode_reward, res_queue
        self.global_net, self.opt = global_net, opt
        self.local_net = network(n_s, n_a)
        self.env = gym.make(env_name).unwrapped

        self.gamma = gamma
        self.max_episode = max_episode
        self.update_global_iter = update_global_iter

    def wrap(self, arr, dtype=np.float32):
        if arr.dtype != dtype:
            arr = arr.astype(dtype)
        return Variable(torch.from_numpy(arr))


    def update_gradients(self, done, states_, states, actions, rewards):
        if done:
            value = 0
        else:
            value = self.local_net.forward(self.wrap(states_[None, :]))[-1].data.numpy()[0, 0]

        targets = []
        for r in rewards[::-1]:
            value = r + self.gamma * value
            targets.append(value)
        targets.reverse()

        loss = self.local_net.loss_func(
            self.wrap(np.vstack(states)),
            self.wrap(np.array(actions), dtype=np.int64) if actions[0].dtype == np.int64 else self.wrap(np.vstack(actions)),
            self.wrap(np.array(targets)[:, None]))

        self.opt.zero_grad()
        loss.backward()
        for lp, gp in zip(self.local_net.parameters(), self.global_net.parameters()):
            gp._grad = lp.grad
        self.opt.step()

        self.local_net.load_state_dict(self.global_net.state_dict())

    def log(self, episode_reward):
        with self.global_episode.get_lock():
            self.global_episode.value += 1
        with self.global_episode_reward.get_lock():
            if self.global_episode_reward.value == 0.:
                self.global_episode_reward.value = episode_reward
            else:
                self.global_episode_reward.value = self.global_episode_reward.value * 0.99 + \
                                                   episode_reward * 0.01
        self.res_queue.put(self.global_episode_reward.value)
        print(
            self.name,
            "Episode:", self.global_episode.value,
            "| Moving Average Reward: %.0f" % self.global_episode_reward.value,
        )

    def run(self):
        total_step = 1
        while self.global_episode.value < self.max_episode:
            state = self.env.reset()
            state_buf, action_buf, reward_buf = [], [], []
            episode_reward = 0.
            while True:
                if self.name == 'w0':
                    self.env.render()
                action = self.local_net.choose_action(self.wrap(state[None, :]))
                state_, reward, done, _ = self.env.step(action)

                episode_reward += reward
                action_buf.append(action)
                state_buf.append(state)
                reward_buf.append(reward)

                if total_step % self.update_global_iter == 0 or done:
                    # sync
                    self.update_gradients(done, state_, state_buf, action_buf, reward_buf)
                    state_buf, action_buf, reward_buf = [], [], []

                    if done:  # done and print information
                        self.log(episode_reward)
                        break
                state = state_
                total_step += 1
        self.res_queue.put(None)


class A3C(object):

    def __init__(self, env_name, network, n_episodes, episodes_per_update, gamma, learning_rate):

        env = gym.make(env_name)
        N_S = env.observation_space.shape[0]
        N_A = env.action_space.n

        global_net = network(N_S, N_A)

        opt = SharedAdamOpt(global_net.parameters(), lr=learning_rate)
        global_episode, global_episode_reward, self.reward_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

        self.rewards = []
        self.workers = [Worker(network, global_net, N_S, N_A, opt, gamma, global_episode, global_episode_reward, \
                               self.reward_queue, i, n_episodes, episodes_per_update, \
                               env_name)
                        for i in range(mp.cpu_count())]

    def train(self):
        [w.start() for w in self.workers]
        while True:
            r = self.reward_queue.get()
            if r is not None:
                self.rewards.append(r)
            else:
                break
        [w.join() for w in self.workers]

    def disply_graph(self):
        import matplotlib.pyplot as plt
        plt.plot(self.rewards)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.show()



if __name__ == "__main__":

    from network import Net
    agent = A3C('CartPole-v0', Net, 40000, 10, 0.9, 0.0001)

    agent.train()
    agent.disply_graph()
