import gym
import numpy as np
import tensorflow as tf

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy

from tensorflow.keras.layers import Flatten, Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

ENV_NAME = 'CartPole-v1'
tf.compat.v1.disable_eager_execution()


class Agent:
    def __init__(self):
        # Get the environment and extract the number of actions.
        self.env = gym.make(ENV_NAME)
        np.random.seed(123)
        self.env.seed(123)
        self.env._max_episode_steps = 10000
        self.nb_actions = self.env.action_space.n

    def build_model(self):
        # Next, we build a very simple model.
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + self.env.observation_space.shape))
        model.add(Dense(40))
        model.add(Activation('relu'))
        model.add(Dense(80))
        model.add(Activation('relu'))
        model.add(Dense(80))
        model.add(Activation('relu'))
        model.add(Dense(40))
        model.add(Activation('relu'))
        model.add(Dense(self.nb_actions))
        model.add(Activation('linear'))
        print(model.summary())

        return model

    def compile(self, save=True):
        # Finally, we configure and compile our agent. You can use every built-in tensorflow.keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=50000, window_length=1)
        policy = BoltzmannQPolicy()
        dqn = DQNAgent(model=self.build_model(), nb_actions=self.nb_actions, memory=memory, nb_steps_warmup=10,
                       target_model_update=1e-2, policy=policy)
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])

        # Okay, now it's time to learn something! We visualize the training here for show, but this
        # slows down training quite a lot. You can always safely abort the training prematurely using
        # Ctrl + C.
        # dqn.fit(self.env, nb_steps=2000, visualize=False, verbose=2)

        # After training is done, we save the final weights.
        # dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME))

        dqn.load_weights('save/infinity/dqn_CartPole-v1_weights.h5f')
        # Finally, evaluate our algorithm for 5 episodes.
        dqn.test(self.env, nb_episodes=5, visualize=True)


if __name__ == '__main__':
    agent = Agent()
    agent.compile()
