import gymnasium as gym
from gymnasium.experimental.wrappers import RescaleActionV0
from tinyrl import load_checkpoint_from_path
import math

policy = load_checkpoint_from_path("pendulum_sac_checkpoint.h")

env = gym.make("Pendulum-v1", render_mode="human")
env = RescaleActionV0(env, -1, 1) # wlog actions are normalized to [-1, 1] in RLtools

while True:
    observation, _ = env.reset()
    finished = False
    while not finished:
        env.render()
        action = math.tanh(policy.evaluate(observation)[0]) # SAC policy outputs mean and std and needs to be squashed by tanh
        observation, reward, terminated, truncated, _ = env.step([action])
        finished = terminated or truncated