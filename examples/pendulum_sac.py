from tinyrl import SAC
import gymnasium as gym

seed = 0xf00d
def env_factory():
    env = gym.make("Pendulum-v1")
    env.reset(seed=seed)
    return env

sac = SAC(env_factory)
state = sac.State(seed)

finished = False
while not finished:
    finished = state.step()