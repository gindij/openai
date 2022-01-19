from discrete_agents.agent import EpsilonGreedyAgent
import numpy as np
import tqdm

class TemporalDifferenceAgent(EpsilonGreedyAgent):

    def __init__(
        self,
        state_space,
        action_space,
        env,
        n_episodes=10000,
        gamma=0.9,
        step_size=0.8,
        ep=1e-1,
        ep_decay_factor=0.8,
        ep_decays=5,
        method="sarsa",
    ):
        super(TemporalDifferenceAgent, self).__init__(state_space, action_space, env)
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.step_size = step_size
        self.n_episodes = n_episodes
        self.step_size = step_size
        self.gamma = gamma
        self.method = method
        self.ep = ep
        self.ep_decay_factor = ep_decay_factor
        self.ep_decay_freq = n_episodes // ep_decays

    def reset(self):
        self.Q = np.zeros((self.n_states, self.n_actions))

    def sarsa_estimate(self, s2, ep):
        # estimate is the action-value of the state and the next action
        a2 = self.act_ep(s2, ep)
        return a2, self.Q[s2, a2]

    def expected_sarsa_estimate(self, s2, ep):
        # estimate is the expectation over possible actions
        probs = np.ones(self.n_actions) * ep / self.n_actions
        probs[np.argmax(self.Q[s2, :])] += 1 - ep
        return None, np.dot(self.Q[s2, :], probs)

    def q_learning_estimate(self, s2):
        # estimate is the maximum over possible actions
        return None, np.max(self.Q[s2, :])

    def action_value_estimate(self, s2, ep):
        if self.method == "sarsa":
            return self.sarsa_estimate(s2, ep)
        if self.method == "esarsa":
            return self.expected_sarsa_estimate(s2, ep)
        if self.method == "qlearn":
            return self.q_learning_estimate(s2)

    def name(self):
        return f"td(0)_{self.method}"

    def learn(self):
        ep = self.ep
        for k in tqdm.tqdm(range(self.n_episodes)):
            s = self.env.reset()
            a = self.act_ep(s, ep)
            done = False
            while not done:
                sn, r, done, _ = self.env.step(a)
                an, est = self.action_value_estimate(sn, ep)
                target = r + self.gamma * est
                # move in the direction of the action value residual
                self.Q[s, a] += self.step_size * (target - self.Q[s, a])
                s = sn
                a = an if an else self.act_ep(sn, ep)
            if k > 0 and k % self.ep_decay_freq == 0:
                ep *= self.ep_decay_factor


class TemporalDifferenceMultiStepAgent(TemporalDifferenceAgent):

    def __init__(
        self,
        state_space,
        action_space,
        env,
        lambd=0.6,
        n_episodes=50000,
        ep=1,
        ep_decay_factor=0.5,
        ep_decays=10,
        gamma=0.99,
        step_size=0.85,
        method="esarsa",
    ):
        super(TemporalDifferenceMultiStepAgent, self).__init__(state_space, action_space, env)
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.step_size = step_size
        self.lambd = lambd
        self.n_episodes = n_episodes
        self.step_size = step_size
        self.gamma = gamma
        assert method in ("sarsa", "esarsa")
        self.method = method
        self.ep = ep
        self.ep_decay_factor = ep_decay_factor
        self.ep_decay_freq = n_episodes // ep_decays

    def reset(self):
        self.Q = np.zeros((self.n_states, self.n_actions))

    def name(self):
        return f"td({self.lambd})_{self.method}"

    def learn(self):
        ep = self.ep
        E = np.zeros((self.n_states, self.n_actions))
        for k in tqdm.tqdm(range(self.n_episodes)):
            # reset eligibility traces to 0 at the beginning of each episode
            E *= 0
            s = self.env.reset()
            a = self.act_ep(s, ep)
            done = False
            while not done:
                sn, r, done, _ = self.env.step(a)
                an, est = self.action_value_estimate(sn, ep)
                update = r + self.gamma * est - self.Q[s, a]
                E[s, a] = 1
                self.Q += self.step_size * update * E
                E *= self.gamma * self.lambd
                s = sn
                a = an if an else self.act_ep(sn, ep)
            if k > 0 and k % self.ep_decay_freq == 0:
                ep *= self.ep_decay_factor
