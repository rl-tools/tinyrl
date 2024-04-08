import gymnasium as gym
from tinyrl import load_checkpoint_from_path

policy = load_checkpoint_from_path("pendulum_sac_checkpoint.h")

env = gym.make("Pendulum-v1", render_mode="human")

while True:
    observation, _ = env.reset()
    finished = False
    while not finished:
        env.render()
        action = policy.evaluate(observation)[:1] # SAC policy outputs mean and std
        observation, reward, terminated, truncated, _ = env.step(action)
        finished = terminated or truncated