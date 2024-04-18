from tinyrl import SAC
import gymnasium as gym
from gymnasium.experimental.wrappers import RescaleActionV0
import math

seed = 0x1337
def env_factory(**kwargs):
    env = gym.make("Pendulum-v1", **kwargs)
    env = RescaleActionV0(env, -1, 1) # wlog actions are normalized to [-1, 1] in RLtools
    env.reset(seed=seed)
    return env

sac = SAC(env_factory)
state = sac.State(seed)

# Training
finished = False
while not finished:
    finished = state.step()

# Save Checkpoint (so it can be loaded by inference.py)
with open("pendulum_sac_checkpoint.h", "w") as f:
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
        action = math.tanh(state.action(observation)[0]) # SAC policy outputs unsquashed actions (for sampling during training), more info: https://github.com/rl-tools/rl-tools/blob/72a59eabf4038502c3be86a4f772bd72526bdcc8/TODO.md?plain=1#L22
        observation, reward, terminated, truncated, _ = env_replay.step(action)
        current_return += reward
        finished = terminated or truncated
    print(f"Return: {current_return}")