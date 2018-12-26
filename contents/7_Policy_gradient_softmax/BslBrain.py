"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Softmax

# reproducible
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

    def _build_net(self):

        inputs = Input(shape=(self.n_features, ))
        l1 = Dense(
            10,
            input_shape=(self.n_features, ),
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(0, 0.3, seed=42),
            bias_initializer=tf.constant_initializer(0.1))(inputs)
        l2 = Dense(
            self.n_actions,
            kernel_initializer=tf.random_normal_initializer(0, 0.3, seed=42),
            bias_initializer=tf.constant_initializer(0.1))(l1)

        reward = Input(shape=(1, ))
        self.opt_model = Model([inputs, reward], [l2])

        def myloss(y_true, y_pred):
            loss = tf.losses.sparse_softmax_cross_entropy(
                labels=tf.to_int64(y_true), logits=y_pred, weights=reward)
            # loss = tf.reduce_mean(loss * reward)
            # tf.nn.sparse_softmax_cross_entropy(y_true, y_pred,
            # weights=reward)
            return loss

        self.opt_model.compile(
            optimizer=tf.train.AdamOptimizer(self.lr), loss=myloss)

        softmax_layer = Softmax()(l2)
        self.eval_model = Model([inputs], [softmax_layer])

    def choose_action(self, observation):
        prob_weights = self.eval_model.predict(observation[np.newaxis, :])

        action = np.random.choice(
            range(prob_weights.shape[1]),
            p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        self.opt_model.train_on_batch(
            [np.vstack(self.ep_obs), discounted_ep_rs_norm],
            np.array(self.ep_as))

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
