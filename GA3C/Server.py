from multiprocessing import Queue

import time
import numpy as np

from GA3C.Config import Config
from GA3C.Environment import Environment
from GA3C.NetworkVP import NetworkVP
from GA3C.Processes import ProcessAgent, ProcessStats

from GA3C.Threads import ThreadPredictor, ThreadTrainer


class Server:
    def __init__(self):
        self.stats = ProcessStats()

        self.training_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.prediction_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)

        self.model = NetworkVP(Config.DEVICE, 'StudentTeacher', Environment().get_num_actions())

        self.training_step = 0
        self.frame_counter = 0

        self.agents = []
        self.predictors = []
        self.trainers = []
        self.rewards = []

    def add_agent(self):
        self.agents.append(
            ProcessAgent(len(self.agents), self.prediction_q, self.training_q, self.stats.episode_log_q))
        self.agents[-1].start()

    def remove_agent(self):
        self.agents[-1].exit_flag.value = True
        self.agents[-1].join(10)
        self.agents.pop()

    def add_predictor(self):
        self.predictors.append(ThreadPredictor(self, len(self.predictors)))
        self.predictors[-1].start()

    def remove_predictor(self):
        self.predictors[-1].exit_flag = True
        self.predictors[-1].join(10)
        self.predictors.pop()

    def add_trainer(self):
        self.trainers.append(ThreadTrainer(self, len(self.trainers)))
        self.trainers[-1].start()

    def remove_trainer(self):
        self.trainers[-1].exit_flag = True
        self.trainers[-1].join(10)
        self.trainers.pop()

    def train_model(self, x_, r_, a_, l_, trainer_id):
        self.model.train(x_, r_, a_, l_, trainer_id)
        self.training_step += 1
        self.frame_counter += x_.shape[0]

        shape = r_.shape[0]

        for i in range(shape):
            r__ = 1.0 if r_[i,0] > 0 else 0.0
            self.rewards.append(float(r__))

        while len(self.rewards) > Config.STAT_ROLLING_MEAN_WINDOW:
            self.rewards = self.rewards[1:]

        #self.rewards = self.rewards[-Config.STAT_ROLLING_MEAN_WINDOW:]

        assert(len(self.rewards) <= Config.STAT_ROLLING_MEAN_WINDOW)
        #self.rewards = np.array(self.rewards, dtype=np.float32)

        tmp = np.array(self.rewards, dtype=np.float32)
        mean = np.mean(tmp)
        self.stats.training_count.value += 1

        if self.stats.training_count.value % Config.TENSORBOARD_UPDATE_FREQUENCY == 0:
            self.model.log(x_, r_, a_, mean)

    def enable_components(self):
        for _ in range(Config.TRAINERS):
            self.add_trainer()
        for _ in range(Config.PREDICTORS):
            self.add_predictor()
        for _ in range(Config.AGENTS):
            self.add_agent()

    def main(self):
        self.stats.start()
        self.enable_components()

        while self.stats.episode_count.value < Config.EPISODES:

            if self.stats.should_save_model.value > 0:
                self.model.save(self.stats.episode_count.value)
                self.stats.should_save_model.value = 0

            time.sleep(0.01)


        # make sure we save the last one
        self.model.save(self.stats.episode_count.value)

        self.stats.exit_flag.value = True
        while self.agents:
            self.remove_agent()
        while self.predictors:
            self.remove_predictor()
        while self.trainers:
            self.remove_trainer()
