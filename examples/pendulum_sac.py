from tinyrl import loop_sac
import gymnasium as gym

def env_factory():
    return gym.make("Pendulum-v1")

loop = loop_sac(env_factory)
finished = False
step = 0

loop_state = loop.LoopState()

loop.init(loop_state, 0)
while not finished:
    finished = loop.step(loop_state)
    step += 1