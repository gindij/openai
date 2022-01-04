def average_reward(agent, n_episodes):
    total_reward = 0
    for i_episode in range(n_episodes):
        state = agent.env.reset()
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done, _ = agent.env.step(action)
            total_reward += reward
    return total_reward / n_episodes

def average_reward_by_number_of_episodes(agent, n_measure, n_episodes_list):
    average_rewards = []
    eps = agent.n_episodes
    for n_episodes in n_episodes_list:
        agent.n_episodes = n_episodes
        agent.reset()
        agent.learn()
        average_rewards.append(average_reward(agent, n_measure))
    agent.n_episodes = eps
    return n_episodes_list, average_rewards

def average_reward_by_step_size(agent, n_episodes, step_sizes):
    average_rewards = []
    ss = agent.step_size
    for step_size in step_sizes:
        agent.step_size = step_size
        agent.reset()
        agent.learn()
        average_rewards.append(average_reward(agent, n_episodes))
    agent.step_size = ss
    return step_sizes, average_rewards

def average_reward_by_epsilon(agent, n_episodes, epsilons):
    average_rewards = []
    ep = agent.ep_orig
    for epsilon in epsilons:
        agent.ep_orig = epsilon
        agent.reset()
        agent.learn()
        average_rewards.append(average_reward(agent, n_episodes))
    agent.ep_orig = ep
    return epsilons, average_rewards

def average_reward_by_discount_factor(agent, n_episodes, discount_factors):
    average_rewards = []
    g = agent.gamma
    for gamma in discount_factors:
        agent.gamma = gamma
        agent.reset()
        agent.learn()
        average_rewards.append(average_reward(agent, n_episodes))
    agent.gamma = g
    return discount_factors, average_rewards
