from abc import ABC, abstractmethod
import numpy as np
import tqdm

class DiscreteEpisodicAgent(ABC):

    def __init__(self, state_space, action_space, env):
        self.state_space = state_space
        self.action_space = action_space
        self.n_states = len(state_space)
        self.n_actions = len(action_space)
        self.env = env

    @abstractmethod
    def act(self, state):
        raise ValueError("Unimplemented")

    @abstractmethod
    def learn(self):
        raise ValueError("Unimplemented")

    @abstractmethod
    def name(self):
        raise ValueError("Unimplemented")

    @abstractmethod
    def reset(self):
        raise ValueError("Unimplemented")


class EpsilonGreedyAgent(DiscreteEpisodicAgent):

    def act(self, state):
        return np.argmax(self.Q[state, :])

    def act_ep(self, state, ep):
        r = np.random.random()
        if r < ep:
            return np.random.choice(self.action_space)
        return np.argmax(self.Q[state, :])
