"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.

The cart pole example. Policy is oscillated.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Softmax
import gym

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000  # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9  # reward discount in TD error
LR_A = 0.001  # learning rate for actor
LR_C = 0.01  # learning rate for critic

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n


class Actor(object):
    def __init__(self, n_features, n_actions, lr=0.001):
        inputs = Input(shape=(n_features, ))
        l1 = Dense(
            20,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1, seed=42),
            bias_initializer=tf.constant_initializer(0.1))(inputs)

        logits = Dense(
            n_actions,
            kernel_initializer=tf.random_normal_initializer(0., .1, seed=42),
            bias_initializer=tf.constant_initializer(0.1),
            name='logits')(l1)

        prob = Softmax(name='prob')(logits)
        reward = Input(shape=(1, ))

        self.train_model = Model([inputs, reward], [logits, prob])

        def myloss(y_true, y_pred):
            loss = tf.losses.sparse_softmax_cross_entropy(
                labels=tf.to_int64(y_true), logits=y_pred, weights=reward)
            return loss

        self.train_model.compile(
            optimizer=tf.train.AdamOptimizer(lr),
            loss={
                'logits': myloss,
                'prob': None
            })

        self.predict_model = Model(inputs, prob)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        a = [[a]]
        cost = self.train_model.train_on_batch([s, td], a)
        print('cost', cost[0])
        return cost[0]

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.predict_model.predict_on_batch(s)
        action = np.random.choice(
            np.arange(probs.shape[1]), p=probs.ravel())  # return a int
        print(s, probs, action)
        return action


class Critic(object):
    def __init__(self, n_features, lr=0.01):
        inputs = Input(shape=(n_features, ))
        x = Dense(
            20,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(
                0., .1, seed=42),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
        )(inputs)
        estimated_reward = Dense(
            1,
            kernel_initializer=tf.random_normal_initializer(
                0., .1, seed=42),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
        )(x)

        self.train_model = Model(inputs=[inputs], outputs=[estimated_reward])

        self.train_model.compile(
            loss='mse', optimizer=tf.train.AdamOptimizer(lr))

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v = self.train_model.predict_on_batch(s)
        v_ = self.train_model.predict_on_batch(s_)
        label = r + GAMMA * v_
        self.train_model.train_on_batch(s, label)

        td_error = np.mean(label - v, axis=1)
        return td_error


actor = Actor(n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(
    n_features=N_F, lr=LR_C
)  # we need a good teacher, so the teacher should learn faster than the actor

for i_episode in range(2):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER: env.render()

        a = actor.choose_action(s)

        s_, r, done, info = env.step(a)

        if done: r = -20

        track_r.append(r)

        td_error = critic.learn(
            s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s, a,
                    td_error)  # true_gradient = grad[logPi(s,a) * td_error]

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break
