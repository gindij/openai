from abc import ABC, abstractmethod
import numpy as np
import tqdm

class DiscreteEpisodicAgent(ABC):

    def __init__(self, state_space, action_space, env):
        self._state_space = state_space
        self._action_space = action_space
        self._policy = None
        self._env = env

    @abstractmethod
    def act(self, state):
        raise ValueError("Unimplemented")

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def policy(self):
        return self._policy

    @property
    def env(self):
        return self._env


class EpsilonGreedyAgent(DiscreteEpisodicAgent, ABC):

    def __init__(
        self,
        state_space,
        action_space,
        env,
        next_state_fn,
        n_updates=10,
        n_episodes=10000,
        ep=1e-1,
        gamma=1
    ):
        super(EpsilonGreedyAgent, self).__init__(state_space, action_space, env)
        n_states = len(state_space)
        n_actions = len(action_space)
        table_dims = (n_states, n_actions)
        self._Q = np.zeros(table_dims)
        self._next_state_fn = next_state_fn
        self._policy = np.ones(table_dims) / n_actions
        self._ep = ep

    def greedify(self):
        for s in self.state_space:
            next_Q_values = [
                self._Q[self._next_state_fn(s, a), a]
                for a in self.action_space
            ]
            max_action_idx = np.argmax(next_Q_values)
            n_actions = len(self.action_space)
            state_policy = np.ones(n_actions) * self._ep / n_actions
            state_policy[max_action_idx] = 1 - self._ep + self._ep / n_actions
            self._policy[s, :] = state_policy

    @abstractmethod
    def update_action_value_function(self):
        raise ValueError("Unimplemented")

    def act(self, state):
        return np.random.choice(self.action_space, p=self.policy[state, :])
