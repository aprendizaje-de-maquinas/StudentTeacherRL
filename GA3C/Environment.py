from queue import Queue

import numpy as np
import scipy.misc as misc

import gym
from GA3C.Config import Config

from PIL import Image


class Experience:
    def __init__(self, state, action, prediction, reward, done):
        self.state = state
        self.action = action
        self.prediction = prediction
        self.reward = reward
        self.done = done


class Environment:
    def __init__(self):

        self.game = gym.make(Config.GAME)

        self.frame_q = Queue(maxsize=1)
        self.previous_state = None
        self.current_state = None
        self.total_reward = 0

        self.cntr = 0
        self.reset()

    def _get_current_state(self):
        if not self.frame_q.full():
            return None  # frame queue is not full yet.
        x_ = np.array(self.frame_q.queue)
        x_ = x_[0]
        return x_

    def _update_frame_q(self, frame):
        if self.frame_q.full():
            self.frame_q.get()

        image = Image.fromarray(frame).resize((Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH), Image.BILINEAR)
        #image.save('{}.jpg'.format(self.cntr))
        #self.cntr += 1
        image = np.array(image, dtype=np.float32) #/ 128.0 - 1.0
        #print(image)
        self.frame_q.put(image)

    def get_num_actions(self):
        return self.game.env.action_space.n

    def reset(self):
        self.total_reward = 0
        self.frame_q.queue.clear()
        self._update_frame_q(self.game.reset())
        self.previous_state = self.current_state = None

    def step(self, action):
        observation, reward, done, _ = self.game.step(action)

        observation = Image.fromarray(observation).resize((Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH), Image.BILINEAR)
        observation = np.array(observation, dtype=np.float32) #/ 128.0 - 1.0

        self.total_reward += reward
        #self._update_frame_q(observation)

        self.previous_state = self.current_state
        self.current_state = self._get_current_state()
        return observation, reward, done
