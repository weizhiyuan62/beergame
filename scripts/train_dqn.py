"""Train and evaluate the baseline DQN experiment."""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beergame.policy.base_stock import BaseStockPolicy
from beergame.policy.dqn import DQNAgent
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env():
    return Env(**ENV_CONFIG)


def make_opponent_policies():
    return {
        0: BaseStockPolicy(target_inventory=100, low=1, high=20),
        2: BaseStockPolicy(target_inventory=100, low=1, high=20),
    }


def opponent_action(policy, state, env, firm_id):
    try:
        return policy.act(state, env=env, firm_id=firm_id)
    except TypeError:
        try:
            return policy.act(state)
        except TypeError:
            return policy.act()


def summarize_result(name, scores, inventory_history, orders_history, demand_history, satisfied_demand_history):
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


def train_dqn_agent(
    env,
    agent,
    num_episodes=NUM_EPISODES,
    max_t=ENV_CONFIG["max_steps"],
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
    opponent_policies=None,
    checkpoint_prefix=None,
    final_model_path=None,
):
    scores = []
    eps = eps_start

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        score = 0

        for _t in range(max_t):
            actions = np.zeros((env.num_firms, 1))
            for firm_id in range(env.num_firms):
                firm_state = state[firm_id].reshape(1, -1)
                if firm_id == agent.firm_id:
                    actions[firm_id] = agent.act(firm_state, eps)
                elif opponent_policies and firm_id in opponent_policies:
                    actions[firm_id] = opponent_action(opponent_policies[firm_id], firm_state, env, firm_id)
                else:
                    actions[firm_id] = random.randint(1, agent.max_order)

            next_state, rewards, done = env.step(actions)
            reward = rewards[agent.firm_id][0]
            agent.step(
                state[agent.firm_id].reshape(1, -1),
                actions[agent.firm_id],
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


def test_dqn_agent(env, agent, num_episodes=TEST_EPISODES, opponent_policies=None):
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
            actions = np.zeros((env.num_firms, 1))
            for firm_id in range(env.num_firms):
                firm_state = state[firm_id].reshape(1, -1)
                if firm_id == agent.firm_id:
                    actions[firm_id] = agent.act(firm_state, epsilon=0.0)
                elif opponent_policies and firm_id in opponent_policies:
                    actions[firm_id] = opponent_action(opponent_policies[firm_id], firm_state, env, firm_id)
                else:
                    actions[firm_id] = random.randint(1, agent.max_order)

            next_state, rewards, done = env.step(actions)
            episode_inventory.append(env.inventory[agent.firm_id][0])
            episode_orders.append(actions[agent.firm_id][0])
            episode_demand.append(env.demand[agent.firm_id][0])
            episode_satisfied_demand.append(env.satisfied_demand[agent.firm_id][0])

            score += rewards[agent.firm_id][0]
            state = next_state

            if done:
                break

        scores.append(score)
        inventory_history.append(episode_inventory)
        orders_history.append(episode_orders)
        demand_history.append(episode_demand)
        satisfied_demand_history.append(episode_satisfied_demand)

        print(f"Test Episode {i_episode}/{num_episodes} | Score: {score:.2f}")

    return scores, inventory_history, orders_history, demand_history, satisfied_demand_history


def run_dqn_experiment(
    name="baseline_dqn",
    agent_class=DQNAgent,
    algorithm="dqn",
    seed_offset=0,
    result_root="results",
    num_episodes=NUM_EPISODES,
    test_episodes=TEST_EPISODES,
):
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
    training_scores = train_dqn_agent(
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
    ) = test_dqn_agent(
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


if __name__ == "__main__":
    summary = run_dqn_experiment()
    print("baseline_dqn summary:", summary)
