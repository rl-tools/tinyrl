from tinyrl import loop_sac
import gymnasium as gym

def test_pendulum_sac():
    def env_factory():
        return gym.make("Pendulum-v1")

    loop = loop_sac(env_factory)
    seed = 1337
    state = loop.State(seed, env_factory)

    finished = False
    while not finished:
        finished = state.step()