import gymnasium as gym

def evaluate_policy(policy, config, env_factory, n_episodes=100, render=False):
    kwargs = {} if not render else {"render_mode": "human"}
    env = env_factory(**kwargs)
    env.reset(seed=config["seed"])
    returns = []
    for episode_i in range(n_episodes):
        observation, _ = env.reset()
        finished = False
        rewards = 0
        while not finished:
            action = policy(observation)
            env.render() if render else None
            observation, reward, terminated, truncated, _ = env.step(action)
            rewards += reward
            finished = terminated or truncated
        returns.append(rewards)
    return returns