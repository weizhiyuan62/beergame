"""Experiment runner for comparing DQN and Double DQN on the beer game."""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from beergame.policy.dqn import DQNAgent, plot_test_results, plot_training_results, test_agent, train_dqn
from beergame.policy.heuristic import BaseStockPolicy
from beergame.policy.improved import DoubleDQNAgent
from beergame.sim.env import Env


ENV_CONFIG = {
    "num_firms": 3,
    "p": [10, 9, 8],
    "h": 0.5,
    "c": 2,
    "initial_inventory": 100,
    "poisson_lambda": 10,
    "max_steps": 100,
}
FIRM_ID = 1
ACTION_SIZE = 20
NUM_EPISODES = 500
TEST_EPISODES = 10
SEED = 42


def set_seed(seed):
    """Set random seeds used by the baseline implementation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env():
    """Create the standard assignment environment."""
    return Env(**ENV_CONFIG)


def summarize_result(name, scores, inventory_history, orders_history, demand_history, satisfied_demand_history):
    """Build report-friendly metrics for one trained policy."""
    summary = {
        "name": name,
        "mean_reward": float(np.mean(scores)),
        "std_reward": float(np.std(scores)),
        "avg_inventory": float(np.mean(inventory_history)),
        "avg_order": float(np.mean(orders_history)),
        "avg_demand": float(np.mean(demand_history)),
        "avg_satisfied_demand": float(np.mean(satisfied_demand_history)),
    }
    if summary["avg_demand"] > 0:
        summary["demand_satisfaction_rate"] = summary["avg_satisfied_demand"] / summary["avg_demand"]
    else:
        summary["demand_satisfaction_rate"] = 0.0
    return summary


def train_and_evaluate(name, agent_class, seed_offset=0, opponent_policies=None):
    """Train one agent class and save its model, plots, and metrics."""
    set_seed(SEED + seed_offset)
    env = make_env()
    agent = agent_class(state_size=3, action_size=ACTION_SIZE, firm_id=FIRM_ID, max_order=ACTION_SIZE)

    scores = train_dqn(
        env,
        agent,
        num_episodes=NUM_EPISODES,
        max_t=ENV_CONFIG["max_steps"],
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        opponent_policies=opponent_policies,
        checkpoint_prefix=f"models/{name}_firm_{FIRM_ID}",
        final_model_path=f"models/{name}_firm_{FIRM_ID}_final.pth",
    )
    plot_training_results(
        scores,
        save_path=f"figures/{name}_training_rewards.png",
        title=f"{name} Training Rewards",
    )

    test_scores, inventory_history, orders_history, demand_history, satisfied_demand_history = test_agent(
        env,
        agent,
        num_episodes=TEST_EPISODES,
        opponent_policies=opponent_policies,
    )
    plot_test_results(
        test_scores,
        inventory_history,
        orders_history,
        demand_history,
        satisfied_demand_history,
        save_path=f"figures/{name}_test_results.png",
        title_prefix=name,
    )

    return scores, summarize_result(
        name,
        test_scores,
        inventory_history,
        orders_history,
        demand_history,
        satisfied_demand_history,
    )


def plot_comparison(training_scores, summaries):
    """Save comparison plots for report figures."""
    plt.figure(figsize=(10, 6))
    for name, scores in training_scores.items():
        window = min(100, len(scores))
        avg_scores = [np.mean(scores[max(0, i - window):i + 1]) for i in range(len(scores))]
        plt.plot(avg_scores, label=name)
    plt.title("DQN vs Double DQN Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/comparison_training_curve.png")
    plt.close()

    names = [summary["name"] for summary in summaries]
    means = [summary["mean_reward"] for summary in summaries]
    stds = [summary["std_reward"] for summary in summaries]

    plt.figure(figsize=(8, 6))
    plt.bar(names, means, yerr=stds, capsize=6)
    plt.title("Test Reward Comparison")
    plt.ylabel("Mean Test Reward")
    plt.tight_layout()
    plt.savefig("figures/comparison_test_scores.png")
    plt.close()


def print_summary(summaries):
    """Print metrics in a compact table-like format."""
    print("\nExperiment summary")
    print("name, mean_reward, std_reward, avg_inventory, avg_order, satisfaction_rate")
    for summary in summaries:
        print(
            f"{summary['name']}, "
            f"{summary['mean_reward']:.2f}, "
            f"{summary['std_reward']:.2f}, "
            f"{summary['avg_inventory']:.2f}, "
            f"{summary['avg_order']:.2f}, "
            f"{summary['demand_satisfaction_rate']:.3f}"
        )


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    plt.rcParams['axes.unicode_minus'] = False

    opponent_policies = {
        0: BaseStockPolicy(target_inventory=100, low=1, high=20),
        2: BaseStockPolicy(target_inventory=100, low=1, high=20),
    }

    experiments = [
        ("baseline_dqn", DQNAgent, 0),
        ("double_dqn", DoubleDQNAgent, 100),
    ]

    training_scores = {}
    summaries = []
    for name, agent_class, seed_offset in experiments:
        scores, summary = train_and_evaluate(
            name,
            agent_class,
            seed_offset=seed_offset,
            opponent_policies=opponent_policies,
        )
        training_scores[name] = scores
        summaries.append(summary)

    plot_comparison(training_scores, summaries)
    print_summary(summaries)


if __name__ == "__main__":
    main()
