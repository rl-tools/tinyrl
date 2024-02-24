from tinyrl import SAC
import gymnasium as gym

def test_pendulum_sac():
    def env_factory():
        return gym.make("Pendulum-v1")

    sac = SAC(env_factory)
    seed = 1337
    state = sac.State(seed, env_factory)

    finished = False
    while not finished:
        finished = state.step()