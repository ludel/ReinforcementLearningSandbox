import os
import pickle
import statistics
from datetime import datetime

import gym
import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout

BASE_PATH = 'save/bipedal'


class Agent:
    def __init__(self, nbr_games, score_requirement, epsilon_decrement, cache_time):
        self.env = gym.make('BipedalWalker-v2')
        self.env.seed(123)

        self.nbr_games = nbr_games
        self.nbr_steps = 400
        self.score_requirement = score_requirement

        self.epsilon = 1
        self.epsilon_decrement = epsilon_decrement

        self.info = {'scores': [], 'games': [], 'constant': [], 'summary': ''}

        self.current_state = None
        self.current_action = self.env.action_space.sample()

        self.action_cursor = 0
        self.cache_time = cache_time

        self.model = self.build_model()
        self.info['constant'] = [self.nbr_games, self.score_requirement, self.epsilon_decrement, self.cache_time]
        self.info['summary'] = self.model.summary()

    @staticmethod
    def build_model():
        model = Sequential()
        model.add(Dense(24, input_dim=24, activation="relu"))
        model.add(Dense(4, activation="tanh"))
        model.compile(loss="mse", optimizer="adam")

        return model

    def export(self):
        dir_path = f'{BASE_PATH}/{datetime.now()}-{self.score_requirement}-{self.nbr_games}'
        os.mkdir(dir_path)

        mean_score = statistics.mean(self.info['scores'])
        self.model.save_weights(f'{dir_path}/{mean_score}.h5')

        pickle.dump(self.info, open(f"{dir_path}/{len(self.info['games'])}.p", 'wb'))

    def random_action(self):
        if self.action_cursor <= self.cache_time:
            self.action_cursor += 1
        else:
            self.action_cursor = 0
            self.current_action = self.env.action_space.sample()

        return self.current_action

    def stupid_action(self):
        if np.random.uniform(0, 1) <= self.epsilon:
            return self.random_action()
        else:
            return self.smart_action()

    def smart_action(self):
        state = np.array(self.current_state).reshape(-1, len(self.current_state))
        action = self.model.predict(state)[0]

        return action

    def session(self, choice_action, render=False):
        states = []
        actions = []
        score = 0
        self.current_state = self.env.reset()

        for _ in range(self.nbr_steps):
            action = choice_action()
            state, reward, done, info = self.env.step(action)
            self.current_state = state

            if render:
                self.env.render()
            if done:
                break

            score += reward
            states.append(state)
            actions.append(action)

        return np.array(states), np.array(actions), score

    def train(self):
        for game_index in range(self.nbr_games):
            self.action_cursor = 0

            states, actions, score = self.session(self.stupid_action, render=False)
            if score >= self.score_requirement:
                self.info['scores'].append(score)
                self.info['games'].append([states, actions])

                self.model.fit(states, actions, verbose=0)
                self.epsilon *= self.epsilon_decrement

                print(f'    -> Game {game_index}: score {score} - epsilon {self.epsilon}')

    def train_by_import(self, save_file):
        with open(save_file, "rb") as file:
            self.info = pickle.load(file)

        for game in self.info['games']:
            self.model.fit(game[0], game[1], verbose=0)
            self.epsilon *= self.epsilon_decrement

    def render_train(self):
        for game in self.info['games']:
            self.env.reset()
            for action in game[1]:
                self.env.step(action)
                self.env.render()

    def test(self):
        for game_index in range(5):
            states, actions, score = self.session(self.smart_action, render=True)
            print(f'test>>> {score}')


if __name__ == '__main__':
    agent = Agent(nbr_games=10000, score_requirement=15, epsilon_decrement=1, cache_time=17)

    print('=> Train')
    agent.train_by_import(f'{BASE_PATH}/2019-12-13 12:20:55.382620-15-100000/15.p')
    # with Pool(5) as p:
    #    p.apply(agent.train)

    # agent.render_train()

    # print('=> Export')
    # agent.export()

    print('=> Test')
    agent.test()
