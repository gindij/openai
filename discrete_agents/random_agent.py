from discrete_agents.agent import DiscreteEpisodicAgent
import numpy as np

class RandomAgent(DiscreteEpisodicAgent):

    def act(self, state):
        return np.random.choice(self._action_space)
