from discrete_agents.agent import EpsilonGreedyAgent
import numpy as np
import tqdm

class TemporalDifferenceAgent(EpsilonGreedyAgent):

    def __init__(
        self,
        state_space,
        action_space,
        env,
        next_state_fn,
        n_episodes=1000,
        ep=1e-1,
        gamma=1,
        step_size=lambda k: 0.5,
        method="sarsa",
    ):
        super(TemporalDifferenceAgent, self).__init__(state_space, action_space, env, next_state_fn, ep=ep)
        n_states = len(state_space)
        n_actions = len(action_space)
        self._policy = np.ones((n_states, n_actions)) / n_actions
        self._step_size = step_size
        self._n_episodes = n_episodes
        self._step_size = step_size
        self._gamma = gamma
        self._method = method
        self.update_action_value_function()
        self.greedify()

    def update_action_value_function(self):
        for k in tqdm.tqdm(range(self._n_episodes)):
            s = self._env.reset()
            a = self.act(s)
            done = False
            while not done:
                sn, r, done, _ = self._env.step(a)
                if self._method == "sarsa":
                    an = self.act(sn)
                    target = r + self._gamma * self._Q[sn, an]
                elif self._method == "qlearn":
                    target = r + self._gamma * max([self._Q[sn, aa] for aa in self.action_space])
                self._Q[s, a] += self._step_size(k) * (target - self._Q[s, a])
                if self._method == "sarsa":
                    s, a = sn, an
                elif self._method == "qlearn":
                    s = sn
