import random
from typing import Tuple

import numpy as np


class EnvGrid:
    def __init__(self):
        self.grid = [
            [1, -1, 10],
            [1, -1, 1],
            [1, 1, 1]
        ]

        self.actions = {'down': [0, 1], 'up': [0, -1], 'right': [1, 0], 'left': [-1, 0]}

        grid_size = len(self.grid) * len(self.grid[0])
        self.Q = [[0, 0, 0, 0] for _ in range(grid_size)]
        self.x = 0
        self.y = 0

        self.epsilon = 0.1  # exploration rate
        self.alpha = 0.9  # learning rate
        self.gamma = 0.96

    @property
    def state(self):
        """
        :return: current state
        """
        return self.y * 3 + self.x

    def reset(self) -> int:
        """
        Reset the environment

        :return: new state
        """
        self.x = 0
        self.y = 0
        return self.state

    def step(self, action) -> Tuple[int, int]:
        """
        Run one timestep of the environment

        :param action:
        :return: current reward and state
        """

        self.x = max(0, min(self.x + action[0], 2))
        self.y = max(0, min(self.y + action[1], 2))

        return self.grid[self.y][self.x], self.state

    def choice_action(self, show=False) -> Tuple[int, list]:
        """
        Choose an action based on epsilon value

        :return: action index and name
        """
        available_actions = list(self.actions.keys())

        if random.uniform(0, 1) < self.epsilon:
            index = random.randrange(0, len(self.actions))
        else:
            index = int(np.argmax(self.Q[self.state]))

        choice = available_actions[index]

        if show:
            print(f'=> {choice}')

        return index, self.actions[choice]

    def q_learning(self, state, state2, action, reward):
        predict = self.Q[state][action]
        target = reward + self.gamma * np.max(self.Q[state2, :])
        self.Q[state][action] = self.Q[state][action] + self.alpha * (target - predict)

    def train(self):
        for index_episode in range(100):
            print(f'Train: {(100 * index_episode) / 100}%')

            state = self.reset()
            action = self.choice_action()

            for _ in range(100):
                reward = self.choice_action()
                state2 = self.state

                self.q_learning(state, state2, action, reward)

                state = state2

                if self.is_done():
                    break

    def show(self):
        print('-' * 9)

        for i_line, line in enumerate(self.grid):
            str_line = []

            for i_col in range(len(line)):
                str_line.append('O' if self.y == i_line and self.x == i_col else 'X')

            print(' | '.join(str_line))

        print('-' * 9, end='\n' * 2)

    def is_done(self):
        return self.grid[self.x][self.y] == 10


if __name__ == '__main__':
    env = EnvGrid()
    env.epsilon = 0.7
    env.alpha = 0.1

    # Train sarsa
    # for i in range(1000):
    #    state = env.reset()
    #    ia, action = env.choice_action()

    #    while not env.is_done():
    #        reward, state1 = env.step(action)
    #        ia1, action1 = env.choice_action()

    #        env.Q[state][ia] = env.Q[state][ia] + env.alpha * (
    #                reward + env.gamma * env.Q[state1][ia1] - env.Q[state][ia]
    #        )

    #       state, action, ia = state1, action1, ia1
    # env.train()

    # Test
    env.epsilon = 0.1
    state_test = env.reset()
    while not env.is_done():
        ia, action_test = env.choice_action(show=True)
        env.step(action_test)
