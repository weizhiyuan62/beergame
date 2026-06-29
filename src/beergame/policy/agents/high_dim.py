"""High-dimensional order-space experiment for the beer game."""

import os
import random
from itertools import product

import numpy as np
import torch

from beergame.policy.agents.dqn import (
    DQNAgent,
    plot_test_results,
    plot_training_results,
    test_agent,
)
from beergame.policy.agents.double_dqn import DoubleDQNAgent
from beergame.policy.heuristic import BaseStockPolicy
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
ORDER_DIM = 3
ORDER_LEVELS = [0, 5, 10, 15, 20]
ORDER_COST_MULTIPLIERS = [1.0, 1.1, 1.25]
MIN_TOTAL_ORDER = 5
MAX_TOTAL_ORDER = 20
ACTION_SPACE = [
    action
    for action in product(ORDER_LEVELS, repeat=ORDER_DIM)
    if MIN_TOTAL_ORDER <= sum(action) <= MAX_TOTAL_ORDER
]
ACTION_SIZE = len(ACTION_SPACE)
NUM_EPISODES = 1000
TEST_EPISODES = 10
SEED = 42


def set_seed(seed):
    """Set random seeds used by the experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env():
    """Create the high-dimensional order-space environment."""
    return Env(
        **ENV_CONFIG,
        order_dim=ORDER_DIM,
        order_cost_multipliers=ORDER_COST_MULTIPLIERS,
    )


def action_to_order_vector(action):
    """Map the DQN's 1-based discrete action to a K-dimensional order vector."""
    return np.asarray(ACTION_SPACE[int(action) - 1], dtype=float)


def scalar_order_to_vector(order):
    """Place a scalar heuristic order in the lowest-cost channel."""
    order_vector = np.zeros(ORDER_DIM, dtype=float)
    order_vector[0] = order
    return order_vector


def opponent_action(policy, state, env, firm_id):
    """Call a non-learning firm's policy with backward-compatible arguments."""
    try:
        return policy.act(state, env=env, firm_id=firm_id)
    except TypeError:
        try:
            return policy.act(state)
        except TypeError:
            return policy.act()


def build_actions(env, agent, state, epsilon, opponent_policies):
    """Build one vector-action matrix for all firms."""
    actions = np.zeros((env.num_firms, ORDER_DIM))
    learning_action = None

    for firm_id in range(env.num_firms):
        firm_state = state[firm_id].reshape(1, -1)
        if firm_id == agent.firm_id:
            learning_action = agent.act(firm_state, epsilon)
            actions[firm_id] = action_to_order_vector(learning_action)
        elif opponent_policies and firm_id in opponent_policies:
            order = opponent_action(opponent_policies[firm_id], firm_state, env, firm_id)
            actions[firm_id] = scalar_order_to_vector(order)
        else:
            order = np.random.choice(ORDER_LEVELS)
            actions[firm_id] = scalar_order_to_vector(order)

    return actions, learning_action


def train_high_dim_dqn(
    env,
    agent,
    num_episodes=500,
    max_t=100,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
    opponent_policies=None,
    checkpoint_prefix=None,
    final_model_path=None,
):
    """Train a DQN-style agent in the high-dimensional order space."""
    scores = []
    eps = eps_start

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        score = 0

        for _t in range(max_t):
            actions, learning_action = build_actions(
                env, agent, state, eps, opponent_policies
            )
            next_state, rewards, done = env.step(actions)

            reward = rewards[agent.firm_id][0]
            agent.step(
                state[agent.firm_id].reshape(1, -1),
                learning_action,
                reward,
                next_state[agent.firm_id].reshape(1, -1),
                done,
            )

            state = next_state
            score += reward

            if done:
                break

        eps = max(eps_end, eps_decay * eps)
        scores.append(score)

        if i_episode % 100 == 0:
            print(
                f"Episode {i_episode}/{num_episodes} | "
                f"Average Score: {np.mean(scores[-100:]):.2f} | "
                f"Epsilon: {eps:.4f}"
            )

        if i_episode % 500 == 0 and checkpoint_prefix:
            agent.save(f"{checkpoint_prefix}_episode_{i_episode}.pth")

    if final_model_path is not None:
        agent.save(final_model_path)

    return scores


def test_high_dim_agent(env, agent, num_episodes=10, opponent_policies=None):
    """Test a trained high-dimensional order-space agent."""
    scores = []
    inventory_history = []
    orders_history = []
    demand_history = []
    satisfied_demand_history = []

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        score = 0
        episode_inventory = []
        episode_orders = []
        episode_demand = []
        episode_satisfied_demand = []

        for _t in range(env.max_steps):
            actions, _learning_action = build_actions(
                env, agent, state, 0.0, opponent_policies
            )
            next_state, rewards, done = env.step(actions)

            episode_inventory.append(env.inventory[agent.firm_id][0])
            episode_orders.append(env.orders[agent.firm_id][0])
            episode_demand.append(env.demand[agent.firm_id][0])
            episode_satisfied_demand.append(env.satisfied_demand[agent.firm_id][0])

            reward = rewards[agent.firm_id][0]
            score += reward
            state = next_state

            if done:
                break

        scores.append(score)
        inventory_history.append(episode_inventory)
        orders_history.append(episode_orders)
        demand_history.append(episode_demand)
        satisfied_demand_history.append(episode_satisfied_demand)

        print(f"Test Episode {i_episode}/{num_episodes} | Score: {score:.2f}")

    return (
        scores,
        inventory_history,
        orders_history,
        demand_history,
        satisfied_demand_history,
    )


def summarize_result(name, scores, inventory_history, orders_history, demand_history, satisfied_demand_history):
    """Build report-friendly metrics for one trained policy."""
    avg_demand = float(np.mean(demand_history))
    avg_satisfied_demand = float(np.mean(satisfied_demand_history))
    return {
        "name": name,
        "mean_reward": float(np.mean(scores)),
        "std_reward": float(np.std(scores)),
        "avg_inventory": float(np.mean(inventory_history)),
        "avg_order": float(np.mean(orders_history)),
        "avg_demand": avg_demand,
        "avg_satisfied_demand": avg_satisfied_demand,
        "demand_satisfaction_rate": (
            avg_satisfied_demand / avg_demand if avg_demand > 0 else 0.0
        ),
    }


def train_and_evaluate(name, agent_class, seed_offset=0, opponent_policies=None):
    """Train one high-dimensional experiment and save model, plots, and metrics."""
    set_seed(SEED + seed_offset)
    env = make_env()
    agent = agent_class(
        state_size=3,
        action_size=ACTION_SIZE,
        firm_id=FIRM_ID,
        max_order=ACTION_SIZE,
    )

    scores = train_high_dim_dqn(
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

    (
        test_scores,
        inventory_history,
        orders_history,
        demand_history,
        satisfied_demand_history,
    ) = test_high_dim_agent(
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
    """Save comparison plots for high-dimensional order-space experiments."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for name, scores in training_scores.items():
        window = min(100, len(scores))
        avg_scores = [np.mean(scores[max(0, i - window):i + 1]) for i in range(len(scores))]
        plt.plot(avg_scores, label=name)
    plt.title("High-Dimensional Order-Space Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/high_dim_comparison_training_curve.png")
    plt.close()

    names = [summary["name"] for summary in summaries]
    means = [summary["mean_reward"] for summary in summaries]
    stds = [summary["std_reward"] for summary in summaries]

    plt.figure(figsize=(8, 6))
    plt.bar(names, means, yerr=stds, capsize=6)
    plt.title("High-Dimensional Test Reward Comparison")
    plt.ylabel("Mean Test Reward")
    plt.tight_layout()
    plt.savefig("figures/high_dim_comparison_test_scores.png")
    plt.close()


def print_summary(summaries):
    """Print metrics in a compact table-like format."""
    print("\nHigh-dimensional order-space experiment summary")
    print(
        f"order_dim={ORDER_DIM}, order_levels={ORDER_LEVELS}, "
        f"total_order_range=[{MIN_TOTAL_ORDER}, {MAX_TOTAL_ORDER}], "
        f"action_size={ACTION_SIZE}"
    )
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
    import matplotlib.pyplot as plt

    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False

    opponent_policies = {
        0: BaseStockPolicy(target_inventory=100, low=0, high=max(ORDER_LEVELS)),
        2: BaseStockPolicy(target_inventory=100, low=0, high=max(ORDER_LEVELS)),
    }
    experiments = [
        ("high_dim_dqn", DQNAgent, 0),
        ("high_dim_double_dqn", DoubleDQNAgent, 100),
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
