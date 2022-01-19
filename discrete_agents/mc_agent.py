from discrete_agents.agent import EpsilonGreedyAgent
import numpy as np
import tqdm

class MonteCarloAgent(EpsilonGreedyAgent):

    def __init__(
        self,
        state_space,
        action_space,
        env,
        n_episodes=10000,
        gamma=1,
        ep=1,
        ep_decay_factor=0.9,
        ep_decays=5,
    ):
        super(MonteCarloAgent, self).__init__(state_space, action_space, env)
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.N = np.zeros((self.n_states, self.n_actions))
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.ep = ep
        self.ep_decay_factor = ep_decay_factor
        self.ep_decay_freq = n_episodes // ep_decays

    def reset(self):
        self.N = np.zeros((self.n_states, self.n_actions))
        self.Q = np.zeros((self.n_states, self.n_actions))

    def episode(self, ep):
        trajectory = []
        done = False
        state = self.env.reset()
        while not done:
            action = self.act_ep(state, ep)
            next_state, reward, done, _ = self.env.step(action)
            trajectory.append((state, action, reward))
            state = next_state
        return trajectory

    def learn(self):
        ep = self.ep
        for k in tqdm.tqdm(range(self.n_episodes)):
            episode = self.episode(ep)
            cum_reward = 0
            reversed_episode = episode[::-1]
            action_value_updates = 0
            for i, (s, a, r) in enumerate(reversed_episode):
                cum_reward = r + self.gamma * cum_reward
                # first visit MC
                earlier_states = set((ss, aa) for ss, aa, _ in reversed_episode[i+1:])
                if (s, a) not in earlier_states:
                    self.N[s, a] += 1
                    self.Q[s, a] += (cum_reward - self.Q[s, a]) / self.N[s, a]
            if k > 0 and k % self.ep_decay_freq == 0:
                ep *= self.ep_decay_factor

    def name(self):
        return "monte_carlo"
