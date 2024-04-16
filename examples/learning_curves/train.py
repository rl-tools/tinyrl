import gymnasium as gym
import numpy as np
import os
import sac

default_config_by_library = {
    "tinyrl": sac.default_config_rltools,
    "sb3": sac.default_config_sb3,
    "cleanrl": sac.default_config_cleanrl
}

if "TINYRL_FULL_RUN" in os.environ:
    print("Using full run config")
    global_config = {
        "n_seeds": 10,
        "n_steps": 1000000,
        "evaluation_interval": 1000,
    }
else:
    global_config = {
        "n_seeds": 10,
        "n_steps": 1000000,
        "evaluation_interval": 1000,
    }

global_config = {
    **global_config,
    "hidden_dim": 256
    # "hidden_dim": 64
}

if __name__ == "__main__":
    import argparse
    import pickle
    configs = {
        # "Pendulum-v1": {
        "Hopper-v4": {
            "tinyrl": list(range(global_config["n_seeds"])),
            "sb3": list(range(global_config["n_seeds"])),
            "cleanrl": list(range(global_config["n_seeds"]))
        }
    }

    flat_configs = []
    flat_config_id = 0
    for environment_name, library_configs in configs.items():
        for library_name, seeds in library_configs.items():
            for seed in seeds:
                config_default = default_config_by_library[library_name]
                config_diff = {"environment_name": environment_name, "library": library_name, "seed": seed}
                config = {**global_config, **config_default, **config_diff}
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
        print("Using TinyRL")
        returns = sac.train_rltools(config)
    elif config["library"] == "sb3":
        print("Using Stable-Baselines3")
        returns = sac.train_sb3(config)
    elif config["library"] == "cleanrl":
        print("Using CleanRL")
        returns = sac.train_cleanrl(config)
    else:
        raise ValueError(f"Unknown library: {config['library']}")
    returns = np.array(returns)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{run_name}.pickle"), 'wb') as f:
        pickle.dump({'returns': returns, 'config': config}, f)
    