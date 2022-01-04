from discrete_agents.agent import DiscreteEpisodicAgent
import numpy as np

class RandomAgent(DiscreteEpisodicAgent):

    def name(self):
        return "random"

    def act(self, state):
        return np.random.choice(self.action_space)

    def learn(self):
        pass
