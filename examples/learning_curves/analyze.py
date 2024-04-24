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
    configs = {}
    input_dir_contents = os.listdir(args.input_dir)
    if len(input_dir_contents) >= 0 and os.path.isdir(os.path.join(args.input_dir, input_dir_contents[0])):
        print(f"The input appears to contain dictionaries (assumed to be runs)")
        run_dir = os.path.join(args.input_dir, sorted(input_dir_contents)[-1])
    else:
        run_dir = args.input_dir
    for input_file in sorted(os.listdir(run_dir)):
        input_path = os.path.join(run_dir, input_file)
        print(f"Loading {input_path}")
        with open(input_path, 'rb') as f:
            run = pickle.load(f)
            config = run["config"]
            returns = run["returns"]
            if config["library"] not in data:
                data[config["library"]] = []
                configs[config["library"]] = []
            data[config["library"]].append(returns)
            configs[config["library"]].append(config)
    
    
    fig, ax = plt.subplots()
    for library, returns in sorted(filter(lambda x: x[0] in library_lookup, data.items()), key=lambda x: list(library_lookup.keys()).index(x[0])):
        print(f"Library: {library}, Runs: {len(returns)}")
        print(f"First Config: {configs[library][0]}")
        returns = np.array(data[library])

        seeds_producing_nan = np.arange(len(returns))[np.any(np.isnan(returns), axis=(1, 2))]
        print(f"Number of seeds producing NaN: {seeds_producing_nan}")


        returns_aggregate = returns.mean(axis=-1)
        print(f"returns aggregate: {returns_aggregate}")

        returns_min = returns_aggregate.min(axis=0)
        returns_mean = returns_aggregate.mean(axis=0)
        returns_median = np.median(returns_aggregate, axis=0)
        returns_std = returns_aggregate.std(axis=0)

        horizontal = range(0, config["n_steps"], config["evaluation_interval"])
        ax.fill_between(horizontal, returns_mean - returns_std, returns_mean + returns_std, alpha=0.1)
        # ax.plot(horizontal, returns_min, label=f"{library_lookup[library]} (min)")
        # ax.plot(horizontal, returns_median, label=f"{library_lookup[library]} (median {returns_median[-1]:.2f}, std {returns_std[-1]:.2f})")
        label = f"{library_lookup[library]} (mean: {returns_mean[-1]:.2f}, std: {returns_std[-1]:.2f})"
        ax.plot(horizontal, returns_mean, label=label)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Returns")

        final_returns = returns[:, -1, :]
        fig_final_return_distribution, ax_final_return_distribution = plt.subplots()
        ax_final_return_distribution.hist(final_returns.ravel(), label=label, bins=10)
        ax_final_return_distribution.legend()
        ax_final_return_distribution.set_title("Final Return Distribution")
        ax_final_return_distribution.set_xlabel("Return")
        ax_final_return_distribution.set_ylabel("Count")
        fig_final_return_distribution.savefig(os.path.join(args.output_dir, f"final_return_distribution_{library}.png"))
    ax.legend()
    fig.savefig(os.path.join(args.output_dir, "learning_curves.png"))




