import gymnasium as gym

def evaluate_policy(policy, config, n_episodes=10, env_factory=None):
    env_replay = gym.make(config["environment_name"]) if env_factory is None else env_factory()
    env_replay.reset(seed=config["seed"])
    returns = []
    for episode_i in range(n_episodes):
        observation, _ = env_replay.reset()
        finished = False
        rewards = 0
        while not finished:
            action = policy(observation)
            observation, reward, terminated, truncated, _ = env_replay.step(action)
            rewards += reward
            finished = terminated or truncated
        returns.append(rewards)
    return returns