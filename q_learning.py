import pickle

import gym
import numpy as np

nb_episode = 1000
nb_step = 100
gamma = 0.96
actions_name = ['Left', 'Down', 'Right', 'Up']


class QLearning:
    def __init__(self, env_name, export_path='data/q_table.pickle'):
        self.env = gym.make(env_name)
        self.env.reset()
        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.epsilon = 0.9
        self.learning_rate = 0.81
        self.export_path = export_path

    def choice_action(self, state):
        return self.env.action_space.sample() if np.random.uniform(0, 1) < self.epsilon else np.argmax(self.Q[state, :])

    def q_learn(self, state, state2, reward, action):
        predict = self.Q[state, action]
        target = reward + gamma * np.max(self.Q[state2, :])
        self.Q[state, action] = self.Q[state, action] + self.learning_rate * (target - predict)

    def sarsa_learn(self, state, state2, reward, action, action2):
        old_value = self.Q[state, action]
        learned_value = reward + gamma * self.Q[state2, action2]
        self.Q[state, action] = (1 - self.learning_rate) * old_value + self.learning_rate * learned_value

    def train(self, learn_function: str):
        for index_episode in range(nb_episode):
            print(f'Train: {(100 * index_episode) / nb_episode}%')

            state = self.env.reset()
            action = self.choice_action(state)

            for _ in range(nb_step):
                state2, reward, done, info = self.take_action(action)

                action2 = self.choice_action(state)
                getattr(self, learn_function, self.q_learn)(state, state2, reward, action, action2)

                state = state2
                action = action2

                if done:
                    break

    def evaluate(self, random_mode=False, render=False):
        nb_steps, nb_rewards = 0, 0

        for _ in range(nb_episode):
            state = self.env.reset()
            steps, reward = 0, 0
            done = False
            states_history = []
            actions_history = []

            while not done:
                if random_mode:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.Q[state, :])

                steps += 1

                state, reward, done, info = self.env.step(action)
                states_history.append(state)
                actions_history.append(action)

                if render and reward:
                    print(f'Win: {reward} - Step: {steps} - States: {states_history} - '
                          f'Actions: {actions_history}')
                    self.env.render()

            nb_rewards += reward
            nb_steps += steps

        print(f"\n=> Results after {nb_episode} episodes in {'random' if random_mode else 'train'} mode:")
        print(f"- Average timesteps per episode: {nb_steps / nb_episode}")
        print(f"- Total reward: {nb_rewards}")

    def take_action(self, action):
        new_state, reward, done, info = self.env.step(action)
        if new_state in [5, 7, 11, 12]:
            reward = -10
        elif new_state == 15:
            reward = 30
        else:
            reward = -5
        return new_state, reward, done, info

    def export_data(self):
        with open(self.export_path, 'wb') as file:
            pickle.dump(self.Q, file)
        print(f'Pickle file exported in {self.export_path}')

    def import_data(self):
        with open(self.export_path, 'rb') as file:
            self.Q = pickle.load(file)
        print(f'Pickle file imported from {self.export_path}')


if __name__ == '__main__':
    q_learning = QLearning('FrozenLake-v0', export_path='data/q_table_taxi.pickle')
    q_learning.train(learn_function='sarsa_learn')
    # q_learning.export_data()
    q_learning.evaluate(random_mode=False, render=True)
