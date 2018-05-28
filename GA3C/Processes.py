from datetime import datetime
from multiprocessing import Process, Queue, Value

import numpy as np
import time

from GA3C.Config import Config
from GA3C.Environment import Environment, Experience

import sys
from queue import Queue as queueQueue


class ProcessAgent(Process):
    def __init__(self, id, prediction_q, training_q, episode_log_q):
        super(ProcessAgent, self).__init__()

        self.id = id
        self.prediction_q = prediction_q
        self.training_q = training_q
        self.episode_log_q = episode_log_q

        self.env = Environment()
        self.num_actions = self.env.get_num_actions()
        self.actions = np.arange(self.num_actions)

        self.discount_factor = Config.DISCOUNT
        # one frame at a time
        self.wait_q = Queue(maxsize=1)
        self.exit_flag = Value('i', 0)

    @staticmethod
    def _accumulate_rewards(experiences, discount_factor, terminal_reward):
        reward_sum = terminal_reward
        for t in reversed(range(0, len(experiences))):
            reward_sum = discount_factor * reward_sum + experiences[t].reward
            experiences[t].reward = reward_sum
        return experiences #[:-1]

    def convert_data(self, experience):

        #print('States', experience.state.shape)
        #print('Actions', len(experience.action))
        #print('Rewards', experience.reward)


        length = experience.state.shape[0]

        x_ = np.zeros((1, Config.TIME_MAX, Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT, 3), np.float)
        x_[0,:length,:,:,:] = experience.state

        a_ = experience.action
        while len(a_) < Config.TIME_MAX:
            a_.append(0)

        a_ = np.eye(self.num_actions)[a_]

        a_ = np.array(a_)
        a_ = np.reshape(a_, [1,-1, self.num_actions])

        r_ = [0.0] * Config.TIME_MAX
        r_[length-1] = experience.reward
        for i in range(length-2, -1, -1):
            r_[i] = r_[i+1] * Config.DISCOUNT

        r_ = np.array(r_)
        r_ = np.reshape(r_, [1,-1])

        return x_, r_, a_, length

    def predict(self, state):
        # put the state in the prediction q
        self.prediction_q.put((self.id, state))
        # wait for the prediction to come back
        p, v = self.wait_q.get()
        #print('out')
        return p, v

    def select_action(self, prediction):
        shape = prediction.shape
        result = []

        for _time in range(shape[0]):
            result.append(np.random.choice(self.actions, p=prediction[_time]))
            if result[-1] == 2: #run action
                break
        return result


    def run_episode(self):
        self.env.reset()
        done = False

        time_count = 0
        reward_sum = 0.0

        while not done:
            # very first few frames
            if self.env.current_state is None:
                self.env.step(0)  # 0 == NOOP
                continue

            prediction, value = self.predict(self.env.current_state)


            action = self.select_action(prediction)
            if action[-1] != 2:
                action = action[:-1]
                action.append(2)

            states = []
            for a in action:
                obs, reward, done = self.env.step(a)
                reward_sum += reward
                states.append(obs)

            states = np.array(states)

            if reward == 1:
                pass
            else:
                reward = -1
            #reward, done = self.env.step(action)
            #reward_sum += reward
            #print(action)
            #print(self.env.previous_state)
            exp = Experience(states, action, prediction, reward, done)

            #if done or time_count == Config.TIME_MAX:

            #updated_exps = ProcessAgent._accumulate_rewards(experiences, self.discount_factor, terminal_reward)
            x_, r_, a_, l_ = self.convert_data(exp)
            if x_ is not None:
                #print(exp.state)
                #print('is none')
                yield x_, r_, a_, l_, reward_sum

            # reset the tmax count
            time_count = 0
            # keep the last experience for the next batch
            #experiences = [experiences[-1]]
            reward_sum = 0.0

            time_count += 1

            self.env.reset()

    def run(self):
        # randomly sleep up to 1 second. helps agents boot smoothly.
        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))

        while self.exit_flag.value == 0:
            total_reward = 0
            total_length = 0
            for x_, r_, a_, l_, reward_sum in self.run_episode():
                total_reward += reward_sum
                total_length += len(r_) + 1  # +1 for last frame that we drop
                self.training_q.put((x_, r_, a_, [l_]))
            self.episode_log_q.put((total_reward, total_length))


class ProcessStats(Process):
    def __init__(self):
        super(ProcessStats, self).__init__()
        self.episode_log_q = Queue(maxsize=100)
        self.episode_count = Value('i', 0)
        self.training_count = Value('i', 0)
        self.should_save_model = Value('i', 0)

        #this is not working........
        self.exit_flag = Value('i', 0)

    def run(self):

        rolling_frame_count = 0
        rolling_reward = 0
        results_q = queueQueue(maxsize=Config.STAT_ROLLING_MEAN_WINDOW)

        self.start_time = time.time()
        while self.exit_flag.value == 0:
            reward, length = self.episode_log_q.get()

            #self.total_frame_count += length
            self.episode_count.value += 1

            rolling_frame_count += length
            rolling_reward += reward

            if results_q.full():
                old_reward, old_length = results_q.get()
                rolling_frame_count -= old_length
                rolling_reward -= old_reward

            results_q.put((reward, length))

            if self.episode_count.value % Config.SAVE_FREQUENCY == 0:
                self.should_save_model.value = 1

            if self.episode_count.value % Config.PRINT_STATS_FREQUENCY == 0:
                print(
                    '[Time: %8d] '
                    '[Episode: %8d Score: %1d] '
                    '[RScore: %10.4f] '
                    % (int(time.time()-self.start_time),
                       self.episode_count.value, reward,
                       rolling_reward / results_q.qsize()))
                sys.stdout.flush()
