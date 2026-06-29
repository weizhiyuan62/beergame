"""Reusable single-algorithm runners and result persistence."""

import json
import random
from pathlib import Path

import numpy as np
import torch

from beergame.policy.dqn import DQNAgent, test_agent, train_dqn
from beergame.policy.double_dqn import DoubleDQNAgent
from beergame.policy.base_stock import BaseStockPolicy
from beergame.policy.high_dim import (
    ACTION_SIZE as HIGH_DIM_ACTION_SIZE,
    ENV_CONFIG as HIGH_DIM_ENV_CONFIG,
    FIRM_ID as HIGH_DIM_FIRM_ID,
    ORDER_DIM,
    ORDER_LEVELS,
    TEST_EPISODES as HIGH_DIM_TEST_EPISODES,
    NUM_EPISODES as HIGH_DIM_NUM_EPISODES,
    make_env as make_high_dim_env,
    summarize_result as summarize_high_dim_result,
    test_high_dim_agent,
    train_high_dim_dqn,
)
from beergame.policy.ppo import PPOAgent, test_ppo_agent, train_ppo
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
    """Set random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env():
    """Create the standard Beer Game environment."""
    return Env(**ENV_CONFIG)


def make_opponent_policies():
    """Use stable base-stock policies for non-learning firms."""
    return {
        0: BaseStockPolicy(target_inventory=100, low=1, high=20),
        2: BaseStockPolicy(target_inventory=100, low=1, high=20),
    }


def summarize_result(name, scores, inventory_history, orders_history, demand_history, satisfied_demand_history):
    """Build report-friendly metrics from one test run."""
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


def save_run_result(
    result_dir,
    name,
    algorithm,
    training_scores,
    test_scores,
    inventory_history,
    orders_history,
    demand_history,
    satisfied_demand_history,
    summary,
    config,
):
    """Persist raw results in a common format for later plotting."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    np.save(result_dir / "training_scores.npy", np.asarray(training_scores, dtype=np.float32))
    np.save(result_dir / "test_scores.npy", np.asarray(test_scores, dtype=np.float32))
    np.savez(
        result_dir / "test_history.npz",
        inventory=np.asarray(inventory_history, dtype=np.float32),
        orders=np.asarray(orders_history, dtype=np.float32),
        demand=np.asarray(demand_history, dtype=np.float32),
        satisfied_demand=np.asarray(satisfied_demand_history, dtype=np.float32),
    )

    metadata = {
        "name": name,
        "algorithm": algorithm,
        "summary": summary,
        "config": config,
    }
    with open(result_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def run_dqn_like(
    name,
    agent_class,
    algorithm,
    result_root="results",
    seed_offset=0,
    num_episodes=NUM_EPISODES,
    test_episodes=TEST_EPISODES,
):
    """Train and test one DQN-style agent, then save raw outputs."""
    set_seed(SEED + seed_offset)
    env = make_env()
    opponent_policies = make_opponent_policies()
    result_dir = Path(result_root) / name
    checkpoint_dir = result_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    agent = agent_class(
        state_size=3,
        action_size=ACTION_SIZE,
        firm_id=FIRM_ID,
        max_order=ACTION_SIZE,
    )
    training_scores = train_dqn(
        env,
        agent,
        num_episodes=num_episodes,
        max_t=ENV_CONFIG["max_steps"],
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        opponent_policies=opponent_policies,
        checkpoint_prefix=str(checkpoint_dir / name),
        final_model_path=str(result_dir / "model.pth"),
    )
    (
        test_scores,
        inventory_history,
        orders_history,
        demand_history,
        satisfied_demand_history,
    ) = test_agent(
        env,
        agent,
        num_episodes=test_episodes,
        opponent_policies=opponent_policies,
    )
    summary = summarize_result(
        name,
        test_scores,
        inventory_history,
        orders_history,
        demand_history,
        satisfied_demand_history,
    )
    save_run_result(
        result_dir,
        name,
        algorithm,
        training_scores,
        test_scores,
        inventory_history,
        orders_history,
        demand_history,
        satisfied_demand_history,
        summary,
        {
            "env": ENV_CONFIG,
            "firm_id": FIRM_ID,
            "action_size": ACTION_SIZE,
            "num_episodes": num_episodes,
            "test_episodes": test_episodes,
            "seed": SEED + seed_offset,
        },
    )
    return summary


def run_ppo(
    name="ppo",
    result_root="results",
    seed_offset=200,
    num_episodes=NUM_EPISODES,
    test_episodes=TEST_EPISODES,
):
    """Train and test PPO, then save raw outputs."""
    set_seed(SEED + seed_offset)
    env = make_env()
    opponent_policies = make_opponent_policies()
    result_dir = Path(result_root) / name
    checkpoint_dir = result_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    agent = PPOAgent(state_size=3, action_size=ACTION_SIZE, firm_id=FIRM_ID)
    training_scores = train_ppo(
        env,
        agent,
        num_episodes=num_episodes,
        max_t=ENV_CONFIG["max_steps"],
        opponent_policies=opponent_policies,
        checkpoint_prefix=str(checkpoint_dir / name),
        final_model_path=str(result_dir / "model.pth"),
    )
    (
        test_scores,
        inventory_history,
        orders_history,
        demand_history,
        satisfied_demand_history,
    ) = test_ppo_agent(
        env,
        agent,
        num_episodes=test_episodes,
        opponent_policies=opponent_policies,
    )
    summary = summarize_result(
        name,
        test_scores,
        inventory_history,
        orders_history,
        demand_history,
        satisfied_demand_history,
    )
    save_run_result(
        result_dir,
        name,
        "ppo",
        training_scores,
        test_scores,
        inventory_history,
        orders_history,
        demand_history,
        satisfied_demand_history,
        summary,
        {
            "env": ENV_CONFIG,
            "firm_id": FIRM_ID,
            "action_size": ACTION_SIZE,
            "num_episodes": num_episodes,
            "test_episodes": test_episodes,
            "seed": SEED + seed_offset,
        },
    )
    return summary


def run_baseline_dqn(**kwargs):
    return run_dqn_like(
        name="baseline_dqn",
        agent_class=DQNAgent,
        algorithm="dqn",
        seed_offset=0,
        **kwargs,
    )


def run_double_dqn(**kwargs):
    return run_dqn_like(
        name="double_dqn",
        agent_class=DoubleDQNAgent,
        algorithm="double_dqn",
        seed_offset=100,
        **kwargs,
    )


def make_high_dim_opponent_policies():
    return {
        0: BaseStockPolicy(target_inventory=100, low=0, high=max(ORDER_LEVELS)),
        2: BaseStockPolicy(target_inventory=100, low=0, high=max(ORDER_LEVELS)),
    }


def run_high_dim_dqn_like(
    name,
    agent_class,
    algorithm,
    result_root="results",
    seed_offset=0,
    num_episodes=HIGH_DIM_NUM_EPISODES,
    test_episodes=HIGH_DIM_TEST_EPISODES,
):
    """Train and test one high-dimensional DQN-style agent."""
    set_seed(SEED + seed_offset)
    env = make_high_dim_env()
    opponent_policies = make_high_dim_opponent_policies()
    result_dir = Path(result_root) / name
    checkpoint_dir = result_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    agent = agent_class(
        state_size=3,
        action_size=HIGH_DIM_ACTION_SIZE,
        firm_id=HIGH_DIM_FIRM_ID,
        max_order=HIGH_DIM_ACTION_SIZE,
    )
    training_scores = train_high_dim_dqn(
        env,
        agent,
        num_episodes=num_episodes,
        max_t=HIGH_DIM_ENV_CONFIG["max_steps"],
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        opponent_policies=opponent_policies,
        checkpoint_prefix=str(checkpoint_dir / name),
        final_model_path=str(result_dir / "model.pth"),
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
        num_episodes=test_episodes,
        opponent_policies=opponent_policies,
    )
    summary = summarize_high_dim_result(
        name,
        test_scores,
        inventory_history,
        orders_history,
        demand_history,
        satisfied_demand_history,
    )
    save_run_result(
        result_dir,
        name,
        algorithm,
        training_scores,
        test_scores,
        inventory_history,
        orders_history,
        demand_history,
        satisfied_demand_history,
        summary,
        {
            "env": HIGH_DIM_ENV_CONFIG,
            "firm_id": HIGH_DIM_FIRM_ID,
            "order_dim": ORDER_DIM,
            "action_size": HIGH_DIM_ACTION_SIZE,
            "num_episodes": num_episodes,
            "test_episodes": test_episodes,
            "seed": SEED + seed_offset,
        },
    )
    return summary


def run_high_dim_dqn(**kwargs):
    return run_high_dim_dqn_like(
        name="high_dim_dqn",
        agent_class=DQNAgent,
        algorithm="high_dim_dqn",
        seed_offset=0,
        **kwargs,
    )


def run_high_dim_double_dqn(**kwargs):
    return run_high_dim_dqn_like(
        name="high_dim_double_dqn",
        agent_class=DoubleDQNAgent,
        algorithm="high_dim_double_dqn",
        seed_offset=100,
        **kwargs,
    )
