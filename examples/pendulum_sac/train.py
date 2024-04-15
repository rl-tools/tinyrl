from tinyrl import SAC
import gymnasium as gym

seed = 0xf00d
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

while True:
    observation, _ = env_replay.reset(seed=seed)
    finished = False
    while not finished:
        env_replay.render()
        action = state.action(observation)
        observation, reward, terminated, truncated, _ = env_replay.step(action)
        finished = terminated or truncated