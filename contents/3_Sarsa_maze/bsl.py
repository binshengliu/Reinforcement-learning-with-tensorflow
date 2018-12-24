"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from maze_env import Maze
import numpy as np
from collections import OrderedDict


class Agent():
    def __init__(self, env, greedy=0.9, learning=0.01, decay=0.9):
        self._env = env
        self.q_table = OrderedDict()
        self.actions = list(range(self._env.n_actions))
        self.learning = learning
        self.decay = decay
        self.greedy = greedy

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = OrderedDict([(i, 0.0) for i in self.actions])

        if np.random.uniform() < self.greedy:
            action = self.state_best_action(state)
        else:
            action = np.random.choice(self.actions)

        print(action)
        return action

    def state_best_action(self, state):
        actions = self.q_table[state]
        best = max(actions.values())
        candidates = [k for k, v in actions.items() if v == best]
        action = np.random.choice(candidates)
        # print('state', state, 'candidates', candidates, 'action', action)
        return action

    def learn(self, state, action, reward, new_state, new_action):
        q_predict = self.q_table[state][action]
        # This step is vitally important for transferring knowledge
        # backward.
        q_target = reward + self.decay * self.q_table[new_state][new_action]

        self.q_table[state][action] += self.learning * (q_target - q_predict)

    def qtable_str(self):
        ret = []
        for key in self.q_table:
            s = ['{}'.format(key)]
            for action in range(self._env.n_actions):
                s.append('{}'.format(self.q_table[key].get(action, 0)))
            s = ' '.join(s)
            ret.append(s)
        return '\n'.join(ret)


def update(agent, env):
    for episode in range(100):
        # initial current_state
        current_state = tuple(env.reset())

        done = False
        current_action = agent.choose_action(current_state)
        while not done:
            # fresh env
            env.render()

            new_state, current_reward, done = env.step(current_action)
            new_state = tuple(new_state)

            new_action = agent.choose_action(new_state)

            agent.learn(
                current_state,
                current_action,
                current_reward,
                new_state,
                new_action,
            )

            current_state = new_state
            current_action = new_action

        print('Episode:', episode)
        print(agent.qtable_str())
        print()
        if episode == 5:
            break
    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    np.random.seed(2)

    env = Maze()
    agent = Agent(env)

    env.after(100, update, agent, env)
    env.mainloop()
