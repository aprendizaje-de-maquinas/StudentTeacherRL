
import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg

from GA3C.Config import Config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


class NetworkVP:

    def __init__(self, device, model_name, num_actions):

        filenames= list(os.listdir ("./logs"))
        
        self.model_name = model_name + '-{}'.format(len(filenames))
        self.num_actions = num_actions

        self.log_eps = Config.LOG_EPSILON
        self.run_action = num_actions - 1
        self.batch_size = Config.PREDICTION_BATCH_SIZE
        self.image_shape = [Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, 3]

        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device(device):

            self._create_graph()

            config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
            config.gpu_options.allow_growth = True

            self.sess = tf.Session(graph=self.graph, config=config)

            self.create_tensorboard()

            self.saver = tf.train.Saver({var.name: var for var in tf.global_variables()}, max_to_keep=0)

            self.sess.run(tf.global_variables_initializer())



    def build_encoder(self):
        _inputs = tf.reshape(self.x, [-1] + self.image_shape)

        _net = vgg.vgg_16(_inputs)
        _layers = _net[1]

        del _layers['vgg_16/fc8'] # the last layer (ie softmax)
        
        _encoding = _layers['vgg_16/fc7'] # the layer before the softmax

        _encoding = _encoding[:,0,0,:]
        _shape = _encoding.get_shape()[-1].value
        _encoding = tf.reshape(_encoding, [self.batch_size, -1, _shape])
        _encoding = tf.transpose(_encoding, [1,0,2]) # shape is now [T_MAX, BATCH_SIZE, _shape]

        self.encodings = _encoding

    def build_decoder(self):

        _policy_dense = tf.layers.Dense(self.num_actions, name='policy')
        _value_dense = tf.layers.Dense(1)


        def sample_fn(_outputs):
            #print('outputs' , _outputs)
            sample = tf.cast(tf.multinomial(_policy_dense(_outputs), 1), tf.float32)
            #print('sample', sample)
            return sample

        def end_fn(sample_ids):
            #ret = tf.reshape(tf.equal(sample_ids, tf.constant(2, dtype=tf.float32)),[-1])
            #print('ret', ret) 
            #return ret
            return tf.tile([False], [self.batch_size])

        def next_fn(sample_ids):
            #print('sample ids' , sample_ids)
            return sample_ids

        _seed_image = self.encodings[0]


        _shape = _seed_image.get_shape()[-1].value
        _lstm = tf.contrib.rnn.LSTMCell(_shape)

        _lstm = tf.contrib.rnn.AttentionCellWrapper(_lstm, 3)


        # change the H to be the encoding of the image
        __initial_state = _lstm.zero_state(self.batch_size, tf.float32)


        print(__initial_state)
        
        #seed initial state with the encoding of the initial game state
        _initial_state = tf.contrib.rnn.LSTMStateTuple(__initial_state[0].c, _seed_image)
        _initial_state = (_initial_state,) + __initial_state[1:]

        #tf.contrib.seq2seq.InferenceHelper
        _sample_helper = CustomHelper(sample_fn,
                                      [1],
                                      tf.float32,
                                      _seed_image,
                                      end_fn,
                                      next_inputs_fn=next_fn
        )


        _sample_decoder = tf.contrib.seq2seq.BasicDecoder(_lstm, _sample_helper, initial_state=_initial_state)


        # _decoded_outputs is a tuple with (rnn_output, sample_ids)
        _decoded_outputs, _final_states, _lengths = tf.contrib.seq2seq.dynamic_decode(
            _sample_decoder,
            output_time_major=True,
            impute_finished=False,
            maximum_iterations=Config.TIME_MAX,
            parallel_iterations=32,
            swap_memory=False,
            scope=None
        )

        #print('decoded outputs', _decoded_outputs)

        _policy_head = tf.nn.softmax(_policy_dense(_decoded_outputs[0]))
        _value_head = _value_dense(_decoded_outputs[0])
        #_value_head  = tf.reshape(_value_dense(_decoded_outputs[0]), [-1, Config.TIME_MAX])

        #print('policy head', _policy_head)
        #print('value head', _value_head)

        self.sampled_results = _decoded_outputs[1]
        self.value_logits = _value_head
        self.policy_sftmx = _policy_head
        self.decodings = _decoded_outputs
        self.lengths = _lengths


    def build_loss(self):

        entropies, advantages, values, imaginations = [], [], [], []
        for i in range(self.batch_size):

            ll = self.lengths[i]

            
            _encodings = self.encodings[:ll,i,:]
            _decodings = self.decodings[0][:ll,i,:]
            a_index = self.action_index[i,:ll,:]
            _policy_head, _value_head = self.policy_sftmx[:ll,i,:], self.value_logits[:ll,i,:]
            _rewards = self.y_r[i,:ll]
  
            _reward_targets = tf.reshape(_rewards, [-1])
            _actions = tf.reshape(a_index, [ll, self.num_actions])

            _policy_head = tf.reshape(_policy_head, [ll, self.num_actions])
            _value_head = tf.reshape(_value_head, [-1])

            selected_action_prob = tf.reduce_sum(_policy_head * _actions, axis=1)

            advantage = tf.log(tf.maximum(selected_action_prob, self.log_eps))
            advantage = tf.multiply(advantage, (_reward_targets - tf.stop_gradient(_value_head)))
            advantages.append(tf.reduce_sum(advantage))
            #advantage = tf.reduce_sum(advantage, axis=0)

            entropy = tf.reduce_sum(tf.log(tf.maximum(_policy_head, self.log_eps)) * _policy_head, axis=1)
            entropy = tf.multiply(-1 * Config.BETA, entropy)
            entropies.append(tf.reduce_sum(entropy))

            _shape = self.encodings.get_shape()[-1].value

            _tmp_encodings = tf.reshape(_encodings, [-1, _shape])
            _tmp_decoder_outputs = tf.reshape(_decodings, [-1, _shape])

            _imagination_cost = tf.reduce_sum(tf.squared_difference(_tmp_decoder_outputs, _tmp_encodings))
            imaginations.append(_imagination_cost)

            _value = 0.5 * tf.reduce_sum(tf.square(_reward_targets -  _value_head), axis=0)

            values.append(_value)



        e = tf.reduce_sum(entropies)
        a = tf.reduce_sum(advantages)
        v = tf.reduce_sum(values)
        im = tf.reduce_sum(imaginations)

        _policy_cost = -(e + a)
        _value_cost = v

        _total_cost = _policy_cost + _value_cost + im


        _optim = tf.train.RMSPropOptimizer( learning_rate=Config.LEARNING_RATE,
                                            decay=Config.RMSPROP_DECAY,
                                            momentum=Config.RMSPROP_MOMENTUM
        )

        with tf.device('gpu:0'):
            _grads, _vars = zip(*_optim.compute_gradients(_total_cost))
            
            #with tf.device('gpu:0'):
            #_clipped = [
            #   None if gradient is None else tf.clip_by_norm(gradient, 5.0)
            #   for gradient in _grads]
            #_clipped = _grads
            _clipped, _ = tf.clip_by_global_norm(_grads, 5.0)
            _train_op = _optim.apply_gradients(zip(_clipped, _vars), global_step=self.global_step)
            #_train_op = _optim.minimize(_total_cost, global_step=self.global_step)

        self.imagination_cost = im
        self.total_cost = _total_cost
        self.train_op = _train_op
        self.optim = _optim
        self.policy_cost = _policy_cost
        self.value_cost = _value_cost
        self.advantage = a
        self.entropy = e

    def _create_graph(self):
        self.x = tf.placeholder(tf.float32, [self.batch_size, None] + self.image_shape)
        self.y_r = tf.placeholder(tf.float32, [self.batch_size, None])
        self.action_index = tf.placeholder(tf.float32, [self.batch_size, None, self.num_actions])
        self.lengths = tf.placeholder(tf.int32, [None])
        self.rolling_reward = tf.placeholder(tf.float32)
        
        self.global_step = tf.Variable(0, trainable=False)

        self.build_encoder()
        self.build_decoder()

        self.build_loss()

    def create_tensorboard(self):
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries.append(tf.summary.scalar("Policy_cost_advantage", self.advantage))
        summaries.append(tf.summary.scalar("Policy_cost_entropy", self.entropy))
        summaries.append(tf.summary.scalar("Total_Policy_cost", self.policy_cost))
        summaries.append(tf.summary.scalar("Value_cost", self.value_cost))
        summaries.append(tf.summary.scalar("Total_cost", self.total_cost))
        summaries.append(tf.summary.scalar("Imagination_cost", self.imagination_cost))
        summaries.append(tf.summary.scalar("Rolling Reward", self.rolling_reward))

        self.summary_op = tf.summary.merge(summaries)
        self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)

    def predict_p_and_v(self, x):
        policy, value = self.sess.run([self.policy_sftmx, self.value_logits], feed_dict={self.x: x})
        return policy, value

    def train(self, x, y_r, a, l, _id):

        feed_dict = {self.x: x, self.y_r: y_r, self.action_index: a, self.lengths:l}
        _ = self.sess.run(self.train_op, feed_dict=feed_dict)

        
    def log(self, x, y_r, a, rr):
        print('logging')
        feed_dict = {self.x: x, self.y_r: y_r, self.action_index: a, self.rolling_reward:rr}
        step, summary = self.sess.run([self.global_step, self.summary_op], feed_dict=feed_dict)
        self.log_writer.add_summary(summary, step)

    def save(self, episode):
        self.saver.save(self.sess, 'checkpoints/%s_%08d' % (self.model_name, episode))


class CustomHelper(tf.contrib.seq2seq.InferenceHelper):

    def __init__(self, sample_fn, sample_shape, sample_dtype, start_inputs, end_fn, next_inputs_fn=None):

        super(CustomHelper, self).__init__(sample_fn, sample_shape, sample_dtype, start_inputs, end_fn, next_inputs_fn)

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        del time, name
        if self._next_inputs_fn is None:
            next_inputs = sample_ids
        else:
            next_inputs = self._next_inputs_fn(outputs)
            finished = self._end_fn(sample_ids)
        return (finished, next_inputs, state)

    
