import json
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_sweep_dataframe(json_path: str) -> pd.DataFrame:
    """Load a sweep JSON file and return a long-format DataFrame with one row per timing measurement."""
    with open(json_path) as f:
        results = json.load(f)

    rows = []
    for r in results:
        overrides = r["overrides"]
        label = overrides.get("override_name", None) if isinstance(overrides, dict) else None
        if label is None:
            label = str(overrides)
        for t in r["times_ms"]:
            rows.append({"condition": label, "time_ms": t, "gpu_name": r["gpu_name"], "num_gpus": r["num_gpus"], "model_dim": r["merged_overrides"]["model_dim"], "n_layer": r["merged_overrides"]["n_layer"]})
    return pd.DataFrame(rows)


def plot_sweep(json_path: str, ax=None):
    """Create a barplot of sweep results with IQR error bars."""
    df = load_sweep_dataframe(json_path)

    print("gpu_name", df["gpu_name"].unique())
    print("num_gpus", df["num_gpus"].unique())
    print("model_dim", df["model_dim"].unique())
    print("n_layer", df["n_layer"].unique())

    keeping = [
        "baseline",
        "+ kernels",
        "+ GNS",
        "+ split heads",
        "dion 0.5",
        "dion 0.25",
        "adamw",
    ]

    df = df[df["condition"].isin(keeping)]

    # Compute relative time normalized by the median of the last condition
    last_condition = df["condition"].iloc[-1]
    baseline_median = df.loc[df["condition"] == last_condition, "time_ms"].median()
    df["relative_time"] = df["time_ms"] / baseline_median

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    plt.rcParams.update({'font.size': 18, 'axes.labelsize': 16, 'axes.titlesize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16})

    sns.barplot(
        data=df,
        x="condition",
        y="relative_time",
        estimator="median",
        errorbar=("pi", 50),  # interquartile range
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Relative time", fontsize=16)
    # ax.set_title(f"Optimizer Step Benchmark: {json_path}")
    ax.set_title(f"Optimizer Step Time vs. AdamW\nphinext 14b, 8x B200s")
    ax.bar_label(ax.containers[0], fmt="%.1f", padding=10)
    plt.xticks(rotation=45, ha="right", fontsize=16)
    
    plt.tight_layout()
    return fig


if False:
    labels = ["baseline", "+ kernels", "+ GNS", "+ split heads", "but dion", "dion .5", "dion .25", "dion .125", "adamw"]
    gpu8 = pd.Series([98, 86, 65, 47, 59, 35, 27, 28, 14], index=labels)
    gpu1 = pd.Series([654, 579, 449, 331, 417, 184, 120, 98, 105], index=labels)

    gpu8 = gpu8 / gpu8.loc["adamw"]
    gpu1 = gpu1 / gpu1.loc["adamw"]


    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax2 = ax1.twinx()

    sns.barplot(x=gpu8.index, y=gpu8.values, ax=ax1, color="steelblue", alpha=0.7, label="GPU8")
    sns.barplot(x=gpu1.index, y=gpu1.values, ax=ax2, color="coral", alpha=0.7, label="GPU1")

    ax1.set_ylabel("GPU8")
    ax2.set_ylabel("GPU1")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot optimizer benchmark sweep results")
    parser.add_argument("json_path", type=str, help="Path to sweep results JSON file")
    args = parser.parse_args()
    fig = plot_sweep(args.json_path)
    fig.savefig("out.svg")
    plt.show()