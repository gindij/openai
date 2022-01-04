from discrete_agents.agent import DiscreteEpisodicAgent
import numpy as np
import tqdm

class TemporalDifferenceAgent(DiscreteEpisodicAgent):

    def __init__(
        self,
        state_space,
        action_space,
        env,
        n_episodes=10000,
        ep=1e-2,
        ep_decay_factor=0.9,
        ep_decay_freq=1000,
        gamma=1,
        step_size=0.8,
        method="sarsa",
    ):
        super(TemporalDifferenceAgent, self).__init__(state_space, action_space, env)
        self.policy = np.ones((self.n_states, self.n_actions)) / self.n_actions
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.step_size = step_size
        self.n_episodes = n_episodes
        self.step_size = step_size
        self.gamma = gamma
        self.method = method
        self.ep = ep
        self.ep_decay_factor = ep_decay_factor
        self.ep_decay_freq = ep_decay_freq

    def reset(self):
        self.policy = np.ones((self.n_states, self.n_actions)) / self.n_actions
        self.Q = np.zeros((self.n_states, self.n_actions))

    def update_policy(self, s):
        best_actions = np.flatnonzero(self.Q[s, :] == np.max(self.Q[s, :]))
        if len(best_actions) == 1:
            a = best_actions[0]
        else:
            # if multiple best actions, choose one at random
            a = np.random.choice(best_actions)
        # update the policy for this state to be epsilon greedy
        self.policy[s, :] = self.ep * np.ones(self.n_actions) / self.n_actions
        self.policy[s, a] = 1 - self.ep + self.ep / self.n_actions

    def sarsa_estimate(self, s2):
        # estimate is the action-value of the state and the next action
        a2 = self.act(s2)
        return self.Q[s2, a2]

    def expected_sarsa_estimate(self, s2):
        # estimate is the expectation over possible actions
        return np.dot(self.Q[s2, :], self.policy[s2, :])

    def q_learning_estimate(self, s2):
        # estimate is the maximum over possible actions
        return np.max(self.Q[s2, :])

    def action_value_estimate(self, s2):
        if self.method == "sarsa":
            return self.sarsa_estimate(s2)
        if self.method == "esarsa":
            return self.expected_sarsa_estimate(s2)
        if self.method == "qlearn":
            return self.q_learning_estimate(s2)

    def name(self):
        return f"temp_diff_{self.method}"

    def act(self, s):
        return np.random.choice(self.action_space, p=self.policy[s, :])

    def learn(self):
        ep = self.ep
        for k in tqdm.tqdm(range(self.n_episodes)):
            s = self.env.reset()
            done = False
            while not done:
                a = self.act(s)
                sn, r, done, _ = self.env.step(a)
                target = r + self.gamma * self.action_value_estimate(sn)
                # move in the direction of the action value residual
                self.Q[s, a] += self.step_size * (target - self.Q[s, a])
                # if there is a change to the value function, update the policy
                if abs(target - self.Q[s, a]) > 0:
                    self.update_policy(s)
                s = sn
            # decrease epsilon
            if k % self.ep_decay_freq == 0:
                ep *= self.ep_decay_factor
