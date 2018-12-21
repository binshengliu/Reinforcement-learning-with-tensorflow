import numpy as np
import time

MAX_EPISODES = 13  # maximum episodes
EPSILON = 0.9
GAMMA = 0.9
ALPHA = 0.1


class Agent():
    def __init__(self, env):
        self._qtable = np.zeros((env.nstates, env.nactions))
        self.env = env
        self.actions = [-1, 1]

    def choose_action(self, state):
        actions_rewards = self._qtable[state]

        if np.random.uniform() > EPSILON or (actions_rewards == 0).all():
            action = np.random.choice(self.actions)
            return self.actions.index(action), action

        action_idx = actions_rewards.argmax()
        return (action_idx, self.actions[action_idx])

    def update_qtable(self, state, action, diff):
        idx = self.actions.index(action)
        self._qtable[state][idx] += diff

    def step(self):
        state = self.env.get_state()
        action_idx, action = self.choose_action(state)
        q_predict = self._qtable[state][action_idx]
        newstate, real_reward = self.env.update_state(action)
        q_target = real_reward + GAMMA * self._qtable[newstate].max()
        self.update_qtable(state, action, ALPHA * (q_target - q_predict))
        print('\r{} {:2d}'.format(self.env.to_str(), action), end='')
        time.sleep(0.1)

    def run(self):
        print('{}'.format(self.env.to_str()), end='')
        for i in range(MAX_EPISODES):
            self.env.reset()
            steps = 0
            while not self.env.is_done():
                self.step()
                steps += 1

            print()
            print('Episode {} Steps: {}'.format(i, steps))
            print(self._qtable)


class Env():
    def __init__(self, nstates):
        self.nstates = nstates
        self.nactions = 2
        self.state = 0

    def reset(self):
        self.state = 0

    def to_str(self):
        s = ['-'] * self.nstates
        s[self.state] = 'o'
        s[-1] = 'T'
        s = ''.join(s)
        return s

    def get_state(self):
        return self.state

    def update_state(self, action):
        self.state += action
        self.state = min(self.nstates - 1, self.state)
        self.state = max(0, self.state)

        reward = 1 if self.state == self.nstates - 1 else 0
        return self.state, reward

    def is_done(self):
        return self.state == self.nstates - 1


def main():
    np.set_printoptions(formatter={'float': lambda x: '{:.6f}'.format(x)})
    np.random.seed(2)  # reproducible
    env = Env(6)
    agent = Agent(env)
    agent.run()


if __name__ == '__main__':
    main()
