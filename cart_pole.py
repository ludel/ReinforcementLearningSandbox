import random
from statistics import mean

import gym
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

number_of_games = 100
number_of_steps = 100
score_requirement = 60


def model_data_preparation(env):
    training_data = []
    accepted_scores = []

    for _ in range(number_of_games):
        score = 0
        games_memory = []
        previous_observation = []

        for __ in range(number_of_steps):
            action = random.randint(0, 1)
            observation, reward, done, info = env.step(action)

            if len(previous_observation) > 0:
                games_memory.append([previous_observation, action])

            previous_observation = observation
            score += reward

            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)

            for game in games_memory:
                output = [0 if game[1] else 1, game[1]]
                training_data.append([game[0], output])

        env.reset()

    return training_data


def build_model(input_size, output_size):
    model = Sequential()

    model.add(Dense(128, input_dim=input_size, activation='relu'))
    model.add(Dense(52, activation='relu'))
    model.add(Dense(output_size, activation='linear'))

    model.compile(loss='mse', optimizer=Adam())

    return model


def train_model(training_data):
    x = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))

    model = build_model(input_size=len(x[0]), output_size=len(y[0]))
    model.fit(x, y, epochs=10, verbose=5)

    return model


def play(env, model):
    scores = []
    choices = []

    for _ in range(number_of_games):
        score = 0
        prev_obs = []

        for __ in range(number_of_steps):
            if len(prev_obs) == 0:
                action = random.randint(0, 1)
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])
            env.render()
            choices.append(action)

            new_obs, reward, done, info = env.step(action)
            prev_obs = new_obs
            score += reward

            if done:
                break

        scores.append(score)
        env.reset()

    print("Average score :", mean(scores))


def main():
    env = gym.make("CartPole-v1")
    env.reset()

    training_data = model_data_preparation(env)
    print(training_data)
    model = train_model(training_data)
    play(env, model)

    env.close()


if __name__ == '__main__':
    main()
