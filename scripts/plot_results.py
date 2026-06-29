"""Plot aggregate figures from saved results.

Edit target_list to choose which result folders are included.
"""

from pathlib import Path
from datetime import datetime
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


target_list = ["baseline_dqn", "double_dqn", "ppo"]
results_root = Path("results")
output_dir = results_root / "summary" / datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_result_dir(name):
    candidate = Path(name)
    if (candidate / "summary.json").exists():
        return candidate

    result_dir = results_root / name
    if (result_dir / "summary.json").exists():
        return result_dir

    run_dirs = [
        path for path in result_dir.iterdir()
        if path.is_dir() and (path / "summary.json").exists()
    ] if result_dir.exists() else []
    if run_dirs:
        return sorted(run_dirs)[-1]

    raise FileNotFoundError(
        f"No result run found for {name}. Expected {result_dir}/<run_time>/summary.json"
    )


def load_result(name):
    result_dir = resolve_result_dir(name)
    summary_path = result_dir / "summary.json"
    training_path = result_dir / "training_scores.npy"
    test_path = result_dir / "test_scores.npy"
    history_path = result_dir / "test_history.npz"

    missing = [
        str(path)
        for path in [summary_path, training_path, test_path, history_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing result files for {name}: {missing}. "
            f"Run scripts/train_{name.replace('baseline_', '')}.py first."
        )

    with open(summary_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "name": name,
        "result_dir": str(result_dir),
        "metadata": metadata,
        "training_scores": np.load(training_path),
        "test_scores": np.load(test_path),
        "history": np.load(history_path),
    }


def moving_average(scores, window=100):
    window = min(window, len(scores))
    return np.asarray([
        np.mean(scores[max(0, i - window):i + 1])
        for i in range(len(scores))
    ])


def plot_training_comparison(results):
    plt.figure(figsize=(10, 6))
    for result in results:
        plt.plot(
            moving_average(result["training_scores"]),
            label=result["name"],
        )
    plt.title("Training Reward Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_training_curve.png")
    plt.close()


def plot_test_score_comparison(results):
    names = [result["name"] for result in results]
    means = [result["metadata"]["summary"]["mean_reward"] for result in results]
    stds = [result["metadata"]["summary"]["std_reward"] for result in results]

    plt.figure(figsize=(8, 6))
    plt.bar(names, means, yerr=stds, capsize=6)
    plt.title("Test Reward Comparison")
    plt.ylabel("Mean Test Reward")
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_test_scores.png")
    plt.close()


def plot_behavior_comparison(results):
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    for result in results:
        history = result["history"]
        name = result["name"]
        axs[0].plot(np.mean(history["inventory"], axis=0), label=name)
        axs[1].plot(np.mean(history["orders"], axis=0), label=name)
        satisfaction_gap = np.mean(history["demand"] - history["satisfied_demand"], axis=0)
        axs[2].plot(satisfaction_gap, label=name)

    axs[0].set_title("Average Inventory")
    axs[0].set_ylabel("Inventory")
    axs[1].set_title("Average Order Quantity")
    axs[1].set_ylabel("Order")
    axs[2].set_title("Average Unsatisfied Demand")
    axs[2].set_xlabel("Time Step")
    axs[2].set_ylabel("Gap")

    for ax in axs:
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_behavior.png")
    plt.close()


def print_summary(results):
    print("name, mean_reward, std_reward, avg_inventory, avg_order, satisfaction_rate")
    for result in results:
        summary = result["metadata"]["summary"]
        print(
            f"{summary['name']}, "
            f"{summary['mean_reward']:.2f}, "
            f"{summary['std_reward']:.2f}, "
            f"{summary['avg_inventory']:.2f}, "
            f"{summary['avg_order']:.2f}, "
            f"{summary['demand_satisfaction_rate']:.3f}"
        )


def main():
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False

    results = [load_result(name) for name in target_list]
    plot_training_comparison(results)
    plot_test_score_comparison(results)
    plot_behavior_comparison(results)
    print_summary(results)
    print(f"Saved figures to {output_dir}")
    print("Used result dirs:")
    for result in results:
        print(f"{result['name']}: {result['result_dir']}")


if __name__ == "__main__":
    main()
