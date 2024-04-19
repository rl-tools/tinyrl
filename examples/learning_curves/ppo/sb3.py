from evaluate_policy import evaluate_policy
import gymnasium as gym
import numpy as np

default_config = {}

def train_sb3(config):
    import os, random
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from stable_baselines3 import PPO
    from stable_baselines3.ppo import MlpPolicy
    from stable_baselines3.common.vec_env import DummyVecEnv
    import torch
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    def env_factory():
        env = gym.make(config["environment_name"])
        env.reset(seed=config["seed"])
        return env
    envs = DummyVecEnv([env_factory for _ in range(config["n_environments"])])
    def policy_factory(obs_dim, action_dim, lr_schedule, **kwargs):
        return MlpPolicy(obs_dim, action_dim, lr_schedule, net_arch=[config["hidden_dim"], config["hidden_dim"]])
    model = PPO(policy_factory, envs,
        learning_rate=config["learning_rate"],
        ent_coef=config["entropy_coefficient"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        batch_size=config["batch_size"],
        n_steps=config["on_policy_runner_steps_per_env"],
    )
    returns = []
    for evaluation_step_i in range(config["n_steps"] // config["evaluation_interval"]):
        model.learn(total_timesteps=config["evaluation_interval"], reset_num_timesteps=False)
        def policy(observation):
            return model.predict(observation, deterministic=True)[0]
        current_returns = evaluate_policy(policy, config)
        print(f"Step {evaluation_step_i * config['evaluation_interval']}/{config['n_steps']}: {np.mean(current_returns)}", flush=True)
        returns.append(current_returns)
    return returns