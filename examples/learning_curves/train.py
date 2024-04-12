import gymnasium as gym
import numpy as np
import os

def evaluate_policy(policy, config, n_episodes=10):
    env_replay = gym.make(config["environment_name"])
    env_replay.reset(seed=config["seed"])
    returns = []
    for episode_i in range(n_episodes):
        observation, _ = env_replay.reset()
        finished = False
        rewards = 0
        while not finished:
            action = policy(observation)
            observation, reward, terminated, truncated, _ = env_replay.step(action)
            rewards += reward
            finished = terminated or truncated
        returns.append(rewards)
    return returns

def scale_action(action, env):
    return action * (env.action_space.high - env.action_space.low) / 2.0 + (env.action_space.high + env.action_space.low) / 2.0


def train_sb3(config):
    import os, random
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from stable_baselines3 import SAC as SB3_SAC
    from stable_baselines3.sac import MlpPolicy
    import torch
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    env = gym.make(config["environment_name"])
    env.reset(seed=config["seed"])
    def policy_factory(obs_dim, action_dim, lr_schedule, **kwargs):
        return MlpPolicy(obs_dim, action_dim, lr_schedule, net_arch=[64, 64])
    model = SB3_SAC(policy_factory, env, learning_rate=1e-3, batch_size=100)
    returns = []
    for evaluation_step_i in range(config["n_steps"] // config["evaluation_interval"]):
        model.learn(total_timesteps=config["evaluation_interval"], reset_num_timesteps=False)
        def policy(observation):
            return model.predict(observation, deterministic=True)[0]
        returns.append(evaluate_policy(policy, config))
    return returns

def train_tinyrl(config, use_python_environment=True):
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

global_config = {
    "n_seeds": 100,
    "n_steps": 20000,
    "evaluation_interval": 100
}

if __name__ == "__main__":
    import argparse
    import pickle
    configs = {
        "Pendulum-v1": {
            "tinyrl": list(range(global_config["n_seeds"])),
            "sb3": list(range(global_config["n_seeds"]))
        }
    }

    flat_configs = []
    flat_config_id = 0
    for environment_name, library_configs in configs.items():
        for library_name, seeds in library_configs.items():
            for seed in seeds:
                config_diff = {"environment_name": environment_name, "library": library_name, "seed": seed}
                config = {**global_config, **config_diff}
                print(f"Config {flat_config_id}: {config_diff}")
                flat_configs.append(config)
                flat_config_id += 1
    print("Number of configs: ", len(flat_configs))
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Pendulum-v1")
    parser.add_argument("--library", type=str, default="tinyrl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=int)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    if args.config is not None:
        config = flat_configs[args.config]
        print(f"Using config {args.config}: {config}")
    else:
        config = {"environment_name": args.environment, "library": args.library, "seed": args.seed, **global_config}
    run_name = f"{config['environment_name']}_{config['library']}_{config['seed']}"
    if config["library"] == "tinyrl":
        returns = train_tinyrl(config)
    elif config["library"] == "sb3":
        returns = train_sb3(config)
    else:
        raise ValueError(f"Unknown library: {config['library']}")
    returns = np.array(returns)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{run_name}.pickle"), 'wb') as f:
        pickle.dump({'returns': returns, 'config': config}, f)
    