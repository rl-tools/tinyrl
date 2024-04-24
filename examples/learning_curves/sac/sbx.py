from evaluate_policy import evaluate_policy
import gymnasium as gym
from gymnasium.experimental.wrappers import RescaleActionV0
import numpy as np

default_config = {}

def train_sbx(config):
    print("Training SBX with config", config, flush=True)
    import os, random
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from sbx import SAC as SBX_SAC
    from sbx.sac.policies import SACPolicy
    import torch
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    def env_factory():
        env = gym.make(config["environment_name"])
        env = RescaleActionV0(env, -1, 1)
        env = gym.wrappers.ClipAction(env)
        env.reset(seed=config["seed"])
        return env
    env = env_factory()
    def policy_factory(obs_dim, action_dim, lr_schedule, **kwargs):
        return SACPolicy(obs_dim, action_dim, lr_schedule, net_arch=[config["hidden_dim"], config["hidden_dim"]])
    model = SBX_SAC(policy_factory, env, learning_starts=config["learning_starts"], learning_rate=config["learning_rate"], batch_size=config["batch_size"], buffer_size=config["n_steps"])
    returns = []
    for evaluation_step_i in range(config["n_steps"] // config["evaluation_interval"]):
        model.learn(total_timesteps=config["evaluation_interval"], reset_num_timesteps=False)
        def policy(observation):
            return model.predict(observation, deterministic=True)[0]
        current_returns = evaluate_policy(policy, config, env_factory)
        print(f"Step {evaluation_step_i * config['evaluation_interval']}/{config['n_steps']}: {np.mean(current_returns)}", flush=True)
        returns.append(current_returns)
    return returns