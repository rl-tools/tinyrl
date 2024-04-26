import numpy as np
import os
import sac
import ppo
from tinyrl import CACHE_PATH

TINYRL_FULL_RUN = "TINYRL_FULL_RUN" in os.environ


global_config = {
    "render": False
}

environment_configs = {
    "SAC": {
        "Pendulum-v1": {
            "n_seeds": 100 if TINYRL_FULL_RUN else 10,
            "n_steps": 20000,
            "evaluation_interval": 100,
            "hidden_dim": 64,
            "learning_rate": 1e-3,
            "learning_starts": 100,
            "batch_size": 100,
        },
        "Hopper-v4": {
            "n_seeds": 30 if TINYRL_FULL_RUN else 10,
            "n_steps": 1000000 if TINYRL_FULL_RUN else 100000,
            "evaluation_interval": 1000,
            "hidden_dim": 64,
            "learning_rate": 3e-4,
            "learning_starts": 10000,
            "batch_size": 128,
        },
        "Ant-v4": {
            "n_seeds": 30 if TINYRL_FULL_RUN else 10,
            "n_steps": 1000000 if TINYRL_FULL_RUN else 100000,
            "evaluation_interval": 10000,
            "hidden_dim": 256,
            "learning_rate": 3e-4,
            "learning_starts": 10000,
            "batch_size": 256,
        }
    },
    "PPO": {
        "Pendulum-v1": {
            "n_seeds": 100 if TINYRL_FULL_RUN else 10,
            "n_steps": 73, # ~ 300k steps
            "evaluation_interval": 1,
            "learning_rate": 1e-3,
            "entropy_coefficient": 0.0,
            "n_epochs": 2,
            "gamma": 0.9,
            "gae_lambda": 0.95,
            "batch_size": 256,
            "hidden_dim": 64,
            "on_policy_runner_steps_per_env": 1024,
            "n_environments": 4,
            "clip_coef": 0.2,
            "norm_advantage": True,
            "initial_action_std": 2.0
        },
        "Hopper-v4": {
            "n_seeds": 30 if TINYRL_FULL_RUN else 10,
            "n_steps": 10000, # ~ 300k steps
            "evaluation_interval": 10,
            "learning_rate": 1e-4,
            "entropy_coefficient": 0.0,
            "n_epochs": 5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "batch_size": 64,
            "hidden_dim": 64,
            "on_policy_runner_steps_per_env": 256,
            "n_environments": 4,
            "clip_coef": 0.2,
            "norm_advantage": True,
            "initial_action_std": 1
        },
    }
}

library_configs = {
    "SAC":{
        # "Pendulum-v1": {
        #     "tinyrl": {**sac.default_config_tinyrl, **environment_configs["SAC"]["Pendulum-v1"]},
        #     "sb3": {**sac.default_config_sb3, **environment_configs["SAC"]["Pendulum-v1"]},
        #     "cleanrl": {**sac.default_config_cleanrl, **environment_configs["SAC"]["Pendulum-v1"]},
        #     "sbx": {**sac.default_config_cleanrl, **environment_configs["SAC"]["Pendulum-v1"]}
        # },
        # "Hopper-v4": {
        #     "tinyrl": {**sac.default_config_tinyrl, **environment_configs["SAC"]["Hopper-v4"]},
        #     "sb3": {**sac.default_config_sb3, **environment_configs["SAC"]["Hopper-v4"]},
        #     "cleanrl": {**sac.default_config_cleanrl, **environment_configs["SAC"]["Hopper-v4"]},
        #     "sbx": {**sac.default_config_sbx, **environment_configs["SAC"]["Hopper-v4"]}
        # },
        # "Ant-v4": {
        #     "tinyrl": {**sac.default_config_tinyrl, **environment_configs["SAC"]["Ant-v4"]},
        #     "sb3": {**sac.default_config_sb3, **environment_configs["SAC"]["Ant-v4"]},
        #     "cleanrl": {**sac.default_config_cleanrl, **environment_configs["SAC"]["Ant-v4"]},
        #     "sbx": {**sac.default_config_sbx, **environment_configs["SAC"]["Ant-v4"]}
        # }
    },
    "PPO":{
        # "Pendulum-v1": {
        #     "tinyrl": {**ppo.default_config_tinyrl, **environment_configs["PPO"]["Pendulum-v1"]},
        #     "sb3": {**ppo.default_config_sb3, **environment_configs["PPO"]["Pendulum-v1"]},
        #     "cleanrl": {**ppo.default_config_cleanrl, **environment_configs["PPO"]["Pendulum-v1"]},
        #     "sbx": {**ppo.default_config_sbx, **environment_configs["PPO"]["Pendulum-v1"]}
        # },
        "Hopper-v4": {
            "tinyrl": {**ppo.default_config_tinyrl, **environment_configs["PPO"]["Hopper-v4"]},
            "sb3": {**ppo.default_config_sb3, **environment_configs["PPO"]["Hopper-v4"]},
            "cleanrl": {**ppo.default_config_cleanrl, **environment_configs["PPO"]["Hopper-v4"]},
            "sbx": {**ppo.default_config_sbx, **environment_configs["PPO"]["Hopper-v4"]}
        },
    }
}

def flatten_configs(algorithm_filter=None, environment_filter=None, library_filter=None):
    flat_configs = []
    flat_config_id = 0
    for algorithm, environment_library_configs in library_configs.items():
        if algorithm_filter is None or algorithm == algorithm_filter:
            for environment_name, current_library_configs in environment_library_configs.items():
                if environment_filter is None or environment_name == environment_filter:
                    for library_name, config in current_library_configs.items():
                        if library_filter is None or library_name == library_filter:
                            for seed in range(config["seed_offset"] if "seed_offset" in config else 0, config["n_seeds"]):
                                config_diff = {"algorithm": algorithm, "environment_name": environment_name, "library": library_name, "seed": seed}
                                config = {**global_config, **config, **config_diff}
                                flat_configs.append(config)
                                flat_config_id += 1
    return flat_configs

if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--list-configs", action="store_true")
    parser.add_argument("--config", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--algorithm", type=str, default=None)
    parser.add_argument("--environment", type=str, default=None)
    parser.add_argument("--library", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    flat_configs = flatten_configs(algorithm_filter=args.algorithm, environment_filter=args.environment, library_filter=args.library)
    print("Number of configs: ", len(flat_configs))
    if args.list_configs:
        for flat_config_id, config in enumerate(flat_configs):
            print(f"Config {flat_config_id}: {config}")
        exit()

    config = flat_configs[args.config]
    print(f"Using config {args.config}: {config}")
    run_name = f"{config['algorithm']}_{config['environment_name']}_{config['library']}_{config['seed']:03d}"
    if config["algorithm"] == "SAC":
        print("Using SAC", flush=True)
        if config["library"] == "tinyrl":
            print("Using TinyRL", flush=True)
            returns = sac.train_tinyrl(config)
        elif config["library"] == "sb3":
            print("Using Stable-Baselines3")
            returns = sac.train_sb3(config)
        elif config["library"] == "cleanrl":
            print("Using CleanRL")
            returns = sac.train_cleanrl(config)
        elif config["library"] == "sbx":
            print("Using SBX")
            returns = sac.train_sbx(config)
        else:
            raise ValueError(f"Unknown library: {config['library']}")
    else:
        print("Using PPO", flush=True)
        if config["library"] == "tinyrl":
            print(f"Using TinyRL PPO (Cache path: {CACHE_PATH})")
            returns = ppo.train_tinyrl(config, verbose=args.verbose)
        elif config["library"] == "sb3":
            print("Using Stable-Baselines3")
            returns = ppo.train_sb3(config)
        elif config["library"] == "cleanrl":
            print("Using CleanRL")
            returns = ppo.train_cleanrl(config)
        elif config["library"] == "sbx":
            print("Using SBX")
            returns = ppo.train_sbx(config)
        else:
            raise ValueError(f"Unknown library: {config['library']}")
    returns = np.array(returns)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{run_name}.pickle"), 'wb') as f:
        pickle.dump({'returns': returns, 'config': config}, f)

    
    