"""Run controlled high-dimensional Beer Game experiments.

The suite fixes training seeds across algorithms and resets the evaluation RNG
before testing. This makes high-dimensional algorithm comparisons less sensitive
to different amounts of random-number consumption during training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from beergame.policy.double_dqn import DoubleDQNAgent
from train_config import default_config_dict, runtime_finish, runtime_start
from train_high_dim_dqn import run_high_dim_dqn_experiment
from train_high_dim_ppo import run_high_dim_ppo_experiment


SUITE_NAME = "controlled_high_dim"
DEFAULT_SEED_OFFSETS = [0, 100, 200]
EVAL_SEED_OFFSET = 10_000


def make_config(
    name,
    algorithm,
    seed_offset,
    num_episodes,
    test_episodes,
    result_root,
    entropy_coef=None,
):
    config = default_config_dict()
    config["training"].update(
        {
            "name": name,
            "algorithm": algorithm,
            "result_root": result_root,
            "num_episodes": num_episodes,
            "test_episodes": test_episodes,
            "seed_offset": seed_offset,
            "eval_seed_offset": EVAL_SEED_OFFSET,
        }
    )
    config["dqn"]["learning_rate"] = 5e-4
    config["ppo"]["learning_rate"] = 1e-4
    config["ppo"]["entropy_coef"] = 0.03 if entropy_coef is None else entropy_coef
    return config


def latest_summary(result_root, name):
    run_dirs = sorted((Path(result_root) / name).glob("*/summary.json"))
    if not run_dirs:
        raise FileNotFoundError(f"No summary found for {name} under {result_root}")
    with open(run_dirs[-1], "r", encoding="utf-8") as f:
        return json.load(f), run_dirs[-1].parent


def run_suite(seed_offsets, num_episodes, test_episodes, result_root):
    experiments = [
        ("high_dim_dqn", "high_dim_dqn", "dqn", None),
        ("high_dim_double_dqn", "high_dim_double_dqn", "double_dqn", None),
        ("high_dim_ppo", "high_dim_ppo", "ppo", None),
        ("high_dim_ppo_no_entropy", "high_dim_ppo", "ppo", 0.0),
    ]
    records = []

    for seed_offset in seed_offsets:
        for name, algorithm, family, entropy_coef in experiments:
            config = make_config(
                name=name,
                algorithm=algorithm,
                seed_offset=seed_offset,
                num_episodes=num_episodes,
                test_episodes=test_episodes,
                result_root=result_root,
                entropy_coef=entropy_coef,
            )
            runtime = runtime_start()
            if family == "dqn":
                summary = run_high_dim_dqn_experiment(config=config, runtime=runtime)
            elif family == "double_dqn":
                summary = run_high_dim_dqn_experiment(
                    agent_class=DoubleDQNAgent,
                    config=config,
                    runtime=runtime,
                )
            else:
                summary = run_high_dim_ppo_experiment(config=config, runtime=runtime)

            metadata, run_dir = latest_summary(result_root, name)
            records.append(
                {
                    "name": name,
                    "algorithm": algorithm,
                    "seed_offset": seed_offset,
                    "effective_seed": metadata["config"]["effective_seed"],
                    "run_dir": str(run_dir),
                    **summary,
                }
            )

    write_outputs(records, Path(result_root) / "_summary")
    return records


def grouped_stats(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record["name"], []).append(record)

    rows = []
    for name, items in grouped.items():
        rewards = np.asarray([item["mean_reward"] for item in items], dtype=float)
        rates = np.asarray([item["demand_satisfaction_rate"] for item in items], dtype=float)
        inventories = np.asarray([item["avg_inventory"] for item in items], dtype=float)
        orders = np.asarray([item["avg_order"] for item in items], dtype=float)
        rows.append(
            {
                "name": name,
                "n": len(items),
                "mean_reward": float(np.mean(rewards)),
                "std_across_seeds": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
                "mean_satisfaction_rate": float(np.mean(rates)),
                "mean_inventory": float(np.mean(inventories)),
                "mean_order": float(np.mean(orders)),
            }
        )
    return sorted(rows, key=lambda row: row["mean_reward"], reverse=True)


def write_outputs(records, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = grouped_stats(records)

    with open(output_dir / "controlled_runs.json", "w", encoding="utf-8") as f:
        json.dump({"records": records, "grouped": stats}, f, ensure_ascii=False, indent=2)

    with open(output_dir / "controlled_runs.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    with open(output_dir / "controlled_grouped.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stats[0].keys())
        writer.writeheader()
        writer.writerows(stats)

    names = [row["name"] for row in stats]
    means = [row["mean_reward"] for row in stats]
    stds = [row["std_across_seeds"] for row in stats]
    plt.figure(figsize=(9, 5))
    plt.bar(names, means, yerr=stds, capsize=5)
    plt.ylabel("Mean test reward across seeds")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "controlled_high_dim_rewards.png")
    plt.close()

    print("Controlled high-dimensional grouped results:")
    for row in stats:
        print(
            f"{row['name']}: reward={row['mean_reward']:.2f} "
            f"+/- {row['std_across_seeds']:.2f}, "
            f"satisfaction={row['mean_satisfaction_rate']:.3f}"
        )
    print(f"Saved controlled summaries to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--test-episodes", type=int, default=20)
    parser.add_argument(
        "--seed-offsets",
        type=int,
        nargs="+",
        default=DEFAULT_SEED_OFFSETS,
    )
    parser.add_argument("--result-root", default=f"results/{SUITE_NAME}")
    return parser.parse_args()


def main():
    args = parse_args()
    run_suite(
        seed_offsets=args.seed_offsets,
        num_episodes=args.episodes,
        test_episodes=args.test_episodes,
        result_root=args.result_root,
    )


if __name__ == "__main__":
    main()
