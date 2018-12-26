"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
import sys

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        if True:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            self.tensorboard = keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=0,
                batch_size=32,
                write_graph=True,
                write_grads=False,
                write_images=False,
                embeddings_freq=0,
                embeddings_layer_names=None,
                embeddings_metadata=None,
                embeddings_data=None)
            self.tensorboard.set_model(self.q_eval_model)

        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        kernal_initializer = tf.random_normal_initializer(0., 0.3, seed=42)
        self.q_eval_model = Sequential([
            Dense(
                10,
                input_shape=(self.n_features, ),
                activation='relu',
                kernel_initializer=kernal_initializer,
                bias_initializer=tf.constant_initializer(0.1)),
            Dense(
                self.n_actions,
                kernel_initializer=kernal_initializer,
                bias_initializer=tf.constant_initializer(0.1),
            ),
        ])
        self.q_eval_model.compile(
            optimizer=tf.train.RMSPropOptimizer(self.lr),
            loss=tf.losses.mean_squared_error)

        self.q_next_model = Sequential([
            Dense(
                10,
                input_shape=(self.n_features, ),
                activation='relu',
                kernel_initializer=kernal_initializer,
                bias_initializer=tf.constant_initializer(0.1),
            ),
            Dense(
                self.n_actions,
                kernel_initializer=kernal_initializer,
                bias_initializer=tf.constant_initializer(0.1),
            ),
        ])

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            # Or just use predict_classes
            action_value = self.q_eval_model.predict(observation)[0]
            action = np.argmax(action_value)
            print(observation, action_value, action)
        else:
            action = np.random.randint(0, self.n_actions)
            print(observation, action, 'r')
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.q_next_model.set_weights(self.q_eval_model.get_weights())
            # print('\ntarget_params_replaced\n')
            # print(self.q_eval_model.get_weights())
            # print(self.q_next_model.get_weights())
            # sys.exit(0)

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(
                self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(
                self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_eval = self.q_eval_model.predict(batch_memory[:, :self.n_features])
        q_next = self.q_next_model.predict(batch_memory[:, -self.n_features:])

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(
            q_next, axis=1)
        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        print('learning', self.learn_step_counter)

        # if self.learn_step_counter == 1:
        #     print(batch_memory[:, :self.n_features], q_target)
        cost = self.q_eval_model.train_on_batch(
            batch_memory[:, :self.n_features], q_target)
        self.tensorboard.on_epoch_end(self.learn_step_counter,
                                      named_logs(self.q_eval_model, [cost]))

        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


def named_logs(model, logs):
    return dict(zip(model.metrics_names, logs))
