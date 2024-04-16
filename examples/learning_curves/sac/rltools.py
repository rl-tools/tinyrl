import os
import gymnasium as gym
from gymnasium.experimental.wrappers import RescaleActionV0
from evaluate_policy import evaluate_policy
import numpy as np

default_config = {}
def env_factory_factory(config, **kwargs):
    def env_factory(**kwargs):
        env = gym.make(config["environment_name"], **kwargs)
        env = RescaleActionV0(env, -1, 1) # wlog actions are normalized to [-1, 1] in RLtools
        env.reset(seed=config["seed"])
        return env
    return env_factory

def train_rltools(config, use_python_environment=True):
    custom_environment = {
        "path": os.path.abspath("../custom_environment"),
        "action_dim": 1,
        "observation_dim": 3,
    }
    env_factory = env_factory_factory(config)
    from tinyrl import SAC
    example_env = env_factory() 
    kwargs = {"STEP_LIMIT": config["n_steps"], "ALPHA": 1, "ACTOR_BATCH_SIZE": 100, "CRITIC_BATCH_SIZE": 100, "OPTIMIZER_EPSILON": 1e-8, "ACTOR_HIDDEN_DIM": config["hidden_dim"], "CRITIC_HIDDEN_DIM": config["hidden_dim"]}
    if use_python_environment:
        sac = SAC(env_factory, force_recompile=not "TINYRL_SKIP_FORCE_RECOMPILE" in os.environ, **kwargs)
    else:
        sac = SAC(custom_environment, **kwargs)
    state = sac.State(config["seed"])
    returns = []
    for step_i in range(config["n_steps"]):
        if step_i % config["evaluation_interval"] == 0:
            returns.append(evaluate_policy(lambda observation: np.tanh(state.action(observation)), config))
        state.step()
    return returns
