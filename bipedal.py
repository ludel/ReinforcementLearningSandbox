import gym
import numpy as np


class Agent:
    def __init__(self):
        self.env = gym.make('BipedalWalker-v2')
        self.Q = np.zeros((self.env.observation_space.shape[0], 10 * 4))
        self.epsilon = 1
        self.alpha = 0.81
        self.gamma = 0.96

        self.nbr_games = 100

    def choice_action(self, state):
        return self.env.action_space.sample() if np.random.uniform(0, 1) < self.epsilon else np.argmax(self.Q[state, :])

    def q_learning(self, state, state2, reward, action):
        predict = self.Q[state, round(action, 1)]
        target = reward + self.gamma * np.max(self.Q[state2, :])
        self.Q[state, action] = self.Q[state, action] + self.alpha * (target - predict)

    def train(self):
        done = False

        for game_index in range(self.nbr_games):
            state = self.env.reset()

            while not done:
                action = self.choice_action(state)
                state2, reward, done, info = self.env.step(action)

                # self.q_learning(state, state2, reward, action)
                state = state2

            print(f'=> Game {game_index}')

    def test(self):
        done = False
        state = self.env.reset()

        self.epsilon = 1

        while not done:
            action = self.choice_action(state)
            state, reward, done, info = self.env.step(action)
            print(reward)
            self.env.render()

        print(done)


if __name__ == '__main__':
    agent = Agent()
    agent.test()
