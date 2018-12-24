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

    def state_best_action_reward(self, state):
        actions = self.q_table.get(state, {})
        if not actions:
            return 0

        return max(actions.values())

    def step(self, state):
        action = self.choose_action(state)
        newstate, reward, done = self._env.step(action)
        newstate = tuple(newstate)

        q_predict = self.q_table[state][action]

        # This step is vitally important for transferring knowledge
        # backward.
        q_target = reward + self.decay * self.state_best_action_reward(
            newstate)

        self.q_table[state][action] += self.learning * (q_target - q_predict)
        return newstate, done

    def qtable_str(self):
        ret = []
        for key in sorted(self.q_table.keys()):
            s = ['{}'.format(key)]
            for action in range(self._env.n_actions):
                s.append('{}'.format(self.q_table[key].get(action, 0)))
            s = ' '.join(s)
            ret.append(s)
        return '\n'.join(ret)


def update(agent):
    for episode in range(100):
        # initial observation
        observation = tuple(env.reset())

        while True:
            # fresh env
            env.render()

            newstate, done = agent.step(observation)

            # swap observation
            observation = newstate

            # break while loop when end of this episode
            if done:
                break

        print('Episode:', episode)
        print(agent.qtable_str())
        print()
        if episode == 20:
            break
    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    np.random.seed(2)

    env = Maze()
    agent = Agent(env)

    env.after(100, update, agent)
    env.mainloop()
