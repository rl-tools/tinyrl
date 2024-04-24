import os
import gymnasium as gym
from gymnasium.experimental.wrappers import RescaleActionV0
from evaluate_policy import evaluate_policy
import numpy as np

default_config = {
    "INITIAL_ACTION_STD": 0.5 # this should be half because for the pendulum the output range is half due to tanh squashing + rescaling
}
def env_factory_factory(config, **kwargs):
    def env_factory(**kwargs):
        env = gym.make(config["environment_name"], **kwargs)
        env = RescaleActionV0(env, -1, 1) # wlog actions are normalized to [-1, 1] in RLtools
        env = gym.wrappers.ClipAction(env)
        env.reset(seed=config["seed"])
        return env
    return env_factory

def train_tinyrl(config, use_python_environment=True, verbose=False):
    custom_environment = {
        "path": os.path.abspath("../custom_environment"),
        "action_dim": 1,
        "observation_dim": 3,
    }
    env_factory = env_factory_factory(config)
    from tinyrl import PPO
    example_env = env_factory() 
    kwargs = {
        "OPTIMIZER_ALPHA": config["learning_rate"],
        "ACTION_ENTROPY_COEFFICIENT": config["entropy_coefficient"],
        "N_EPOCHS": config["n_epochs"],
        "GAMMA": config["gamma"],
        "LAMBDA": config["gae_lambda"],
        "BATCH_SIZE": config["batch_size"],
        "ACTOR_HIDDEN_DIM": config["hidden_dim"],
        "CRITIC_HIDDEN_DIM": config["hidden_dim"],
        "ON_POLICY_RUNNER_STEPS_PER_ENV": config["on_policy_runner_steps_per_env"],
        "N_ENVIRONMENTS": config["n_environments"],
        "EPSILON_CLIP": config["clip_coef"],
        "NORMALIZE_ADVANTAGE": config["norm_advantage"],
        "STEP_LIMIT": config["n_steps"],
        "OPTIMIZER_EPSILON": 1e-8, # PyTorch default
        **default_config
    }
    interface_name = str(config["seed"])
    if use_python_environment:
        ppo = PPO(env_factory, enable_evaluation=True, interface_name=interface_name, force_recompile=not "TINYRL_SKIP_FORCE_RECOMPILE" in os.environ, verbose=verbose, **kwargs)
    else:
        ppo = PPO(custom_environment, enable_evaluation=False, interface_name=interface_name, verbose=verbose, **kwargs)
    state = ppo.State(config["seed"])
    returns = []
    for step_i in range(config["n_steps"]):
        if step_i % config["evaluation_interval"] == 0:
            current_returns = evaluate_policy(lambda observation: state.action(observation), config, env_factory=env_factory)
            print(f"Step {step_i}/{config['n_steps']}: {np.mean(current_returns)}", flush=True)
            returns.append(current_returns)
        state.step()
    return returns
