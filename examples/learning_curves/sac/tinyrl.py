import os
import gymnasium as gym
from gymnasium.experimental.wrappers import RescaleActionV0
from evaluate_policy import evaluate_policy
import numpy as np

default_config = {
}
def env_factory_factory(config, **kwargs):
    def env_factory(**kwargs):
        env = gym.make(config["environment_name"], **kwargs)
        env = RescaleActionV0(env, -1, 1) # wlog actions are normalized to [-1, 1] in RLtools
        env = gym.wrappers.ClipAction(env)
        env.reset(seed=config["seed"])
        return env
    return env_factory

def train_tinyrl(config, use_python_environment=True):
    custom_environment = {
        "path": os.path.abspath("../custom_environment"),
        "action_dim": 1,
        "observation_dim": 3,
    }
    env_factory = env_factory_factory(config)
    from tinyrl import SAC
    example_env = env_factory() 
    kwargs = {
        "STEP_LIMIT": config["n_steps"],
        "ALPHA": 1,
        "ACTOR_BATCH_SIZE": config["batch_size"],
        "CRITIC_BATCH_SIZE": config["batch_size"],
        "OPTIMIZER_ALPHA": config["learning_rate"],
        "OPTIMIZER_EPSILON": 1e-8, # PyTorch default
        "ACTOR_HIDDEN_DIM": config["hidden_dim"],
        "CRITIC_HIDDEN_DIM": config["hidden_dim"],
        "N_WARMUP_STEPS": config["learning_starts"],
    }
    interface_name = str(config["seed"])
    if use_python_environment:
        sac = SAC(env_factory, enable_evaluation=False, interface_name=interface_name, force_recompile=not "TINYRL_SKIP_FORCE_RECOMPILE" in os.environ, **kwargs)
    else:
        sac = SAC(custom_environment, interface_name=interface_name, **kwargs)
    state = sac.State(config["seed"])
    returns = []
    for step_i in range(config["n_steps"]):
        if step_i % config["evaluation_interval"] == 0:
            current_returns = evaluate_policy(lambda observation: np.tanh(state.action(observation)), config, env_factory)
            print(f"Step {step_i}/{config['n_steps']}: {np.mean(current_returns)}", flush=True)
            returns.append(current_returns)
        state.step()
    return returns
