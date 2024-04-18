from tinyrl import TD3
import gymnasium as gym
from gymnasium.experimental.wrappers import RescaleActionV0

seed = 0x1337
def env_factory(**kwargs):
    env = gym.make("Pendulum-v1", **kwargs)
    env = RescaleActionV0(env, -1, 1) # wlog actions are normalized to [-1, 1] in RLtools
    env.reset(seed=seed)
    return env

td3 = TD3(env_factory)
state = td3.State(seed)

# Training
finished = False
while not finished:
    finished = state.step()

# Save Checkpoint (so it can be loaded by inference.py)
with open("pendulum_td3_checkpoint.h", "w") as f:
    f.write(state.export_policy())

# Inference
env_replay = env_factory(render_mode="human")
env_replay.reset(seed=seed)

while True:
    observation, _ = env_replay.reset()
    finished = False
    current_return = 0
    while not finished:
        env_replay.render()
        action = state.action(observation)
        observation, reward, terminated, truncated, _ = env_replay.step(action)
        current_return += reward
        finished = terminated or truncated
    print(f"Return: {current_return}")