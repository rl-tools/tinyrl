from evaluate_policy import evaluate_policy
import gymnasium as gym
from gymnasium.experimental.wrappers import RescaleActionV0
import numpy as np

default_config = {}

def train_sb3(config):
    import os, random
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from stable_baselines3 import SAC as SB3_SAC
    from stable_baselines3.sac import MlpPolicy
    import torch
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    def env_factory(**kwargs):
        env = gym.make(config["environment_name"], **kwargs)
        env = RescaleActionV0(env, -1, 1)
        env = gym.wrappers.ClipAction(env)
        env.reset(seed=config["seed"])
        return env
    env = env_factory()
    def policy_factory(obs_dim, action_dim, lr_schedule, **kwargs):
        return MlpPolicy(obs_dim, action_dim, lr_schedule, net_arch=[config["hidden_dim"], config["hidden_dim"]], activation_fn=torch.nn.ReLU)
    model = SB3_SAC(policy_factory, env, learning_starts=config["learning_starts"], learning_rate=config["learning_rate"], batch_size=config["batch_size"], buffer_size=config["n_steps"])
    returns = []
    for evaluation_step_i in range(0, config["n_steps"], config["evaluation_interval"]):
        def policy(observation):
            return model.predict(observation, deterministic=True)[0]
        current_returns = evaluate_policy(policy, config, env_factory, render=config["render"] and evaluation_step_i>=0)
        print(f"Step {evaluation_step_i}: {np.mean(current_returns)}", flush=True)
        returns.append(current_returns)
        model.learn(total_timesteps=config["evaluation_interval"], reset_num_timesteps=False)
    return returns