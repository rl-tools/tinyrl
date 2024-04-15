from tinyrl import SAC
import gymnasium as gym

seed = 0x1337
def env_factory():
    env = gym.make("Pendulum-v1")
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
env_replay = gym.make("Pendulum-v1", render_mode="human")
low, high = env_replay.action_space.low, env_replay.action_space.high
half_range, offset = (high - low) / 2, (high + low) / 2

while True:
    observation, _ = env_replay.reset(seed=seed)
    finished = False
    while not finished:
        env_replay.render()
        action = state.action(observation) * half_range + offset # wlog actions are normalized to [-1, 1] in RLtools
        observation, reward, terminated, truncated, _ = env_replay.step(action)
        finished = terminated or truncated