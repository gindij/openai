import gym
from discrete_agents import RandomAgent, MonteCarloAgent, TemporalDifferenceAgent
from next_state import frozen_lake_v0_next_state

def measure(agent, n_episodes):
    total_reward = 0
    total_steps = 0
    wins = 0
    for i_episode in range(n_episodes):
        observation = agent._env.reset()
        done = False
        steps = 0
        while not done:
            action = agent.act(observation)
            observation, reward, done, _ = agent._env.step(action)
            total_reward += reward
            steps += 1
        if reward == 1:
            wins += 1
            total_steps += steps
    return (
        total_reward / n_episodes,
        (total_steps / wins if wins > 0 else -1)
    )

env1 = gym.make('FrozenLake-v1')
env2 = gym.make('FrozenLake-v1')
env3 = gym.make('FrozenLake-v1')

state_space = list(range(env1.observation_space.n))
action_space = list(range(env1.action_space.n))

random_agent = RandomAgent(
    state_space=state_space,
    action_space=action_space,
    env=env1,
)
mc_agent = MonteCarloAgent(
    state_space=state_space,
    action_space=action_space,
    env=env2,
    next_state_fn=frozen_lake_v0_next_state,
    n_episodes=1000,
    n_updates=10,
    ep=1e-2,
    gamma=1,
)
td_agent = TemporalDifferenceAgent(
    state_space=state_space,
    action_space=action_space,
    env=env3,
    next_state_fn=frozen_lake_v0_next_state,
    n_episodes=10000,
    ep=1e-2,
    gamma=1,
    step_size=lambda k: 1 / (k + 1),
    method="qlearn"
)

N_EPISODES = 1000
print(measure(random_agent, N_EPISODES))
print(measure(mc_agent, N_EPISODES))
print(measure(td_agent, N_EPISODES))

env1.close()
env2.close()
env3.close()
