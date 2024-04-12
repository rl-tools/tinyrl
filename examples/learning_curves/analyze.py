import argparse
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input-dir", type=str, required=True)
    argparser.add_argument("--output-dir", type=str, required=True)
    args = argparser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = {}
    input_dir_contents = os.listdir(args.input_dir)
    if len(input_dir_contents) >= 0 and os.path.isdir(os.path.join(args.input_dir, input_dir_contents[0])):
        print(f"The input appears to contain dictionaries (assumed to be runs)")
        run_dir = os.path.join(args.input_dir, sorted(input_dir_contents)[-1])
    else:
        run_dir = args.input_dir
    for input_file in os.listdir(run_dir):
        input_path = os.path.join(run_dir, input_file)
        print(f"Loading {input_path}")
        with open(input_path, 'rb') as f:
            run = pickle.load(f)
            config = run["config"]
            returns = run["returns"]
            if config["library"] not in data:
                data[config["library"]] = []
            data[config["library"]].append(returns)
    
    
    returns_tinyrl = np.array(data["tinyrl"])
    returns_sb3 = np.array(np.array(data["sb3"]))

    returns_tinyrl_aggregate = returns_tinyrl.mean(axis=-1)
    returns_sb3_aggregate = returns_sb3.mean(axis=-1)

    returns_tinyrl_mean = returns_tinyrl_aggregate.mean(axis=0)
    returns_sb3_mean = returns_sb3_aggregate.mean(axis=0)
    returns_tinyrl_std = returns_tinyrl_aggregate.std(axis=0)
    returns_sb3_std = returns_sb3_aggregate.std(axis=0)

    horizontal = range(0, config["n_steps"], config["evaluation_interval"])
    plt.fill_between(horizontal, returns_tinyrl_mean - returns_tinyrl_std, returns_tinyrl_mean + returns_tinyrl_std, alpha=0.1)
    plt.plot(horizontal, returns_tinyrl_mean, label="TinyRL")
    plt.fill_between(horizontal, returns_sb3_mean - returns_sb3_std, returns_sb3_mean + returns_sb3_std, alpha=0.1)
    plt.plot(horizontal, returns_sb3_mean, label="Stable Baselines3")
    plt.xlabel("Steps")
    plt.ylabel("Returns")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "learning_curves.png"))


