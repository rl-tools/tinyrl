from tinyrl import SAC
import os 

seed = 0xf00d

custom_environment = {
    "path": os.path.dirname(os.path.abspath(__file__)),
    "action_dim": 1,
    "observation_dim": 3,
}

sac = SAC(custom_environment)
state = sac.State(seed)

# Training
finished = False
while not finished:
    finished = state.step()

# Inference
import gymnasium as gym
env_replay = gym.make("Pendulum-v1", render_mode="human")

while True:
    observation, _ = env_replay.reset(seed=seed)
    finished = False
    while not finished:
        env_replay.render()
        action = state.action(observation)
        action *= env_replay.env.env.env.max_torque # wlog. in RLtools the action is normalized to be in [-1, 1]
        observation, reward, terminated, truncated, _ = env_replay.step(action)
        finished = terminated or truncated



