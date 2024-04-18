import argparse
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

library_lookup = {
    "sb3": "Stable Baselines3",
    "cleanrl": "CleanRL",
    "tinyrl": "TinyRL / RLtools",
    "sbx": "SBX",
}

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
    
    
    plt.figure()
    for library, returns in sorted(data.items(), key=lambda x: list(library_lookup.keys()).index(x[0])):
        print(f"Library: {library}, Runs: {len(returns)}")
        returns = np.array(data[library])

        returns_aggregate = returns.mean(axis=-1)

        returns_mean = returns_aggregate.mean(axis=0)
        returns_median = np.median(returns_aggregate, axis=0)
        returns_std = returns_aggregate.std(axis=0)

        horizontal = range(0, config["n_steps"], config["evaluation_interval"])
        plt.fill_between(horizontal, returns_mean - returns_std, returns_mean + returns_std, alpha=0.1)
        # plt.plot(horizontal, returns_mean, label=library_lookup[library])
        plt.plot(horizontal, returns_median, label=library_lookup[library])
        plt.xlabel("Steps")
        plt.ylabel("Returns")
        plt.legend()
    plt.savefig(os.path.join(args.output_dir, "learning_curves.png"))


