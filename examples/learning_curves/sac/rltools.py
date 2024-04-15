import os
import gymnasium as gym
from evaluate_policy import evaluate_policy

def scale_action(action, env):
    return action * (env.action_space.high - env.action_space.low) / 2.0 + (env.action_space.high + env.action_space.low) / 2.0


default_config = {}

def train_rltools(config, use_python_environment=True):
    custom_environment = {
        "path": os.path.abspath("../custom_environment"),
        "action_dim": 1,
        "observation_dim": 3,
    }

    from tinyrl import SAC
    example_env = gym.make(config["environment_name"])
    kwargs = {"STEP_LIMIT": config["n_steps"], "ALPHA": 1, "ACTOR_BATCH_SIZE": 100, "CRITIC_BATCH_SIZE": 100, "OPTIMIZER_EPSILON": 1e-8}
    if use_python_environment:
        def env_factory():
            env = gym.make(config["environment_name"])
            env.reset(seed=config["seed"])
            return env
        sac = SAC(env_factory, force_recompile=True, **kwargs)
    else:
        sac = SAC(custom_environment, **kwargs)
    state = sac.State(config["seed"])
    returns = []
    for step_i in range(config["n_steps"]):
        if step_i % config["evaluation_interval"] == 0:
            returns.append(evaluate_policy(lambda observation: scale_action(state.action(observation), example_env), config))
        state.step()
    return returns
