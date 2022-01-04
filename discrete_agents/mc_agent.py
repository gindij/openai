from discrete_agents.agent import DiscreteEpisodicAgent
import numpy as np
import tqdm

class MonteCarloAgent(DiscreteEpisodicAgent):

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
        super(MonteCarloAgent, self).__init__(state_space, action_space, env, ep=ep)
        self.gamma = gamma
        self.n_updates = n_updates
        self.n_episodes = n_episodes
        self.N = np.zeros((self.n_states, self.n_actions))
        self.Q = np.zeros((self.n_states, self.n_actions))

    def reset(self):
        self.N = np.zeros((len(state_space), len(action_space)))
        self.Q = np.zeros((self.n_states, self.n_actions))

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
        episodes = [self.episode() for _ in range(self.n_episodes)]
        for episode in episodes:
            cum_reward = 0
            reversed_episode = episode[::-1]
            for i, (s, a, r) in enumerate(reversed_episode):
                cum_reward = self.gamma * cum_reward + r
                # first visit MC
                earlier_states = set((ss, aa) for ss, aa, _ in reversed_episode[i+1:])
                if (s, a) not in earlier_states:
                    self.N[s, a] += 1
                    self.Q[s, a] += (cum_reward - self.Q[s, a]) / self.N[s, a]

    def greedify(self):
        best_actions = np.argmax(self.Q, axis=1)
        n_states = len(self.state_space)
        n_actions = len(self.action_space)
        self.policy = self.ep * np.ones((n_states, n_actions)) / n_actions
        self.policy[self.state_space, best_actions] = 1 - self.ep + self.ep / n_actions

    def name(self):
        return "monte_carlo"

    def learn(self):
        for _ in range(self.n_updates):
            self.update_action_value_function()
            self.greedify()

    def act(self, state):
        return np.random.choice(self.action_space, p=self.policy[state, :])
