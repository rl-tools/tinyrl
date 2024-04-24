from tinyrl import SAC
import os, time

def test_custom_environment():
    seed = 0xf00d

    custom_environment = {
        "path": os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "custom_environment")),
        "action_dim": 1,
        "observation_dim": 3,
    }

    sac = SAC(custom_environment, interface_name="test_custom_environment")
    state = sac.State(seed)

    # Training
    start = time.time()
    finished = False
    while not finished:
        finished = state.step()
    end = time.time()
    print(f"Training time: {end - start}")
    if "TINYRL_FORCE_MKL" in os.environ:
        assert(end - start < 10.0)

    # Inference
    import gymnasium as gym
    env_replay = gym.make("Pendulum-v1")

    observation, _ = env_replay.reset(seed=seed)
    action = state.action(observation)



