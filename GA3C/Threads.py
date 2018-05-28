import time
from threading import Thread
import numpy as np
from GA3C.Config import Config


class ThreadPredictor(Thread):
    def __init__(self, server, id):
        super(ThreadPredictor, self).__init__()
        self.setDaemon(True)

        self.id = id
        self.server = server
        self.exit_flag = False

    def run(self):
        ids = np.zeros(Config.PREDICTION_BATCH_SIZE, dtype=np.uint16)

        states = np.zeros(
            (Config.PREDICTION_BATCH_SIZE, 1, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, 3),
            dtype=np.float32)

        while not self.exit_flag:
            ids[0], states[0] = self.server.prediction_q.get()

            size = 1
            while size < Config.PREDICTION_BATCH_SIZE and not self.exit_flag:
                ids[size], states[size] = self.server.prediction_q.get()
                size += 1

            batch = states[:size]
            p, v = self.server.model.predict_p_and_v(batch)

            p = np.transpose(p, [1,0,2])
            v = np.transpose(v, [1,0,2])

            for i in range(size):
                if ids[i] < len(self.server.agents):
                    self.server.agents[ids[i]].wait_q.put((p[i], v[i]))


class ThreadTrainer(Thread):
    def __init__(self, server, id):
        super(ThreadTrainer, self).__init__()
        self.setDaemon(True)

        self.id = id
        self.server = server
        self.exit_flag = False

    def run(self):
        while not self.exit_flag:
            batch_size = 0
            while batch_size < Config.PREDICTION_BATCH_SIZE and not self.exit_flag:
                x_, r_, a_, l_ = self.server.training_q.get()
                if batch_size == 0:
                    x__ = x_; r__ = r_; a__ = a_; l__=l_
                else:
                    #print(r__.shape, r_.shape)
                    x__ = np.concatenate((x__, x_))
                    r__ = np.concatenate((r__, r_))
                    a__ = np.concatenate((a__, a_))
                    l__ = np.concatenate((l__, l_))
                batch_size += x_.shape[0]

            x__ = x__[:Config.PREDICTION_BATCH_SIZE]
            r__ = r__[:Config.PREDICTION_BATCH_SIZE]
            a__ = a__[:Config.PREDICTION_BATCH_SIZE]
            l__ = l__[:Config.PREDICTION_BATCH_SIZE]


            self.server.train_model(x__, r__, a__, l__,self.id)
