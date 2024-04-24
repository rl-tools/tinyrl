from tinyrl import SAC
import gymnasium as gym
from gymnasium.experimental.wrappers import RescaleActionV0

def test_multi_interface():
    seed = 0xf00d
    def env_factory():
        env = gym.make("Pendulum-v1")
        env = RescaleActionV0(env, -1, 1) # wlog actions are normalized to [-1, 1] in RLtools
        env = gym.wrappers.ClipAction(env)
        env.reset(seed=seed)
        return env

    sac = SAC(env_factory, interface_name="test_pendulum_sac")
    sac2 = SAC(env_factory, interface_name="test_pendulum_sac2", STEP_LIMIT=13337)

    state = sac.State(seed)

    finished = False
    while not finished:
        finished = state.step()

    state = sac2.State(seed)

    finished = False
    while not finished:
        finished = state.step()



if __name__ == "__main__":
    test_multi_interface()