import gymnasium as gym
import numpy as np
import os
import sac

default_config_by_library = {
    "tinyrl": sac.default_config_tinyrl,
    "sb3": sac.default_config_sb3,
    "cleanrl": sac.default_config_cleanrl,
    "sbx": sac.default_config_sbx
}

full_run = "TINYRL_FULL_RUN" in os.environ


environment_configs = {
    "SAC": {
        "Pendulum-v1": {
            "n_seeds": 100 if full_run else 10,
            "n_steps": 20000,
            "evaluation_interval": 100,
            "hidden_dim": 64,
            "learning_rate": 1e-3,
            "learning_starts": 100,
            "batch_size": 100,
        },
        "Hopper-v4": {
            "n_seeds": 30 if full_run else 10,
            "n_steps": 1000000 if full_run else 100000,
            "evaluation_interval": 1000,
            "hidden_dim": 64,
            "learning_rate": 3e-4,
            "learning_starts": 10000,
            "batch_size": 128,
        }
    }
}

library_configs = {
    "SAC":{
        "Pendulum-v1": {
            "tinyrl": {**sac.default_config_tinyrl, **environment_configs["SAC"]["Pendulum-v1"]},
            "sb3": {**sac.default_config_sb3, **environment_configs["SAC"]["Pendulum-v1"]},
            "cleanrl": {**sac.default_config_cleanrl, **environment_configs["SAC"]["Pendulum-v1"]},
            "sbx": {**sac.default_config_cleanrl, **environment_configs["SAC"]["Pendulum-v1"]}
        },
        "Hopper-v4": {
            "tinyrl": {**sac.default_config_tinyrl, **environment_configs["SAC"]["Hopper-v4"]},
            "sb3": {**sac.default_config_sb3, **environment_configs["SAC"]["Hopper-v4"]},
            "cleanrl": {**sac.default_config_cleanrl, **environment_configs["SAC"]["Hopper-v4"]}
        }
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
                                config = {**config, **config_diff}
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
    if config["library"] == "tinyrl":
        print("Using TinyRL")
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
    returns = np.array(returns)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{run_name}.pickle"), 'wb') as f:
        pickle.dump({'returns': returns, 'config': config}, f)
    