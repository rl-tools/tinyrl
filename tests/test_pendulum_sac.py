from tinyrl import SAC
import gymnasium as gym
from gymnasium.experimental.wrappers import RescaleActionV0

def test_pendulum_sac():
    seed = 0xf00d
    def env_factory():
        env = gym.make("Pendulum-v1")
        env = RescaleActionV0(env, -1, 1) # wlog actions are normalized to [-1, 1] in RLtools
        env = gym.wrappers.ClipAction(env)
        env.reset(seed=seed)
        return env

    sac = SAC(env_factory, interface_name="test_pendulum_sac")
    state = sac.State(seed)

    finished = False
    while not finished:
        finished = state.step()