from discrete_agents.agent import EpsilonGreedyAgent
import numpy as np
import tqdm

class MonteCarloAgent(EpsilonGreedyAgent):

    def __init__(
        self,
        state_space,
        action_space,
        next_state_fn,
        env,
        n_episodes=1000,
        n_updates=15,
        ep=1e-1,
        gamma=1,
    ):
        super(MonteCarloAgent, self).__init__(state_space, action_space, env, next_state_fn, ep=ep)
        self._gamma = gamma
        self._n_updates = n_updates
        self._n_episodes = n_episodes
        self._N = np.zeros((len(state_space), len(action_space)))
        self.iterate_policy()

    def episode(self):
        trajectory = []
        done = False
        state = self.env.reset()
        while not done:
            action = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            trajectory.append((state, action, reward))
            state = next_state
        return trajectory

    def update_action_value_function(self):
        episodes = [self.episode() for _ in range(self._n_episodes)]
        for episode in episodes:
            cum_reward = 0
            reversed_episode = episode[::-1]
            for i, (s, a, r) in enumerate(reversed_episode):
                cum_reward = self._gamma * cum_reward + r
                # first visit MC
                earlier_states = set((ss, aa) for ss, aa, _ in reversed_episode[i+1:])
                if (s, a) not in earlier_states:
                    self._N[s, a] += 1
                    self._Q[s, a] += (cum_reward - self._Q[s, a]) / self._N[s, a]

    def iterate_policy(self):
        for _ in tqdm.tqdm(range(self._n_updates)):
            self.update_action_value_function()
            self.greedify()
