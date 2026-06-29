"""Train and evaluate the high-dimensional DQN experiment."""

from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beergame.policy.base_stock import BaseStockPolicy
from beergame.policy.dqn import DQNAgent
from beergame.policy.high_dim import (
    ACTION_SIZE,
    ENV_CONFIG,
    FIRM_ID,
    NUM_EPISODES,
    ORDER_DIM,
    ORDER_LEVELS,
    SEED,
    TEST_EPISODES,
    action_to_order_vector,
    make_env,
    scalar_order_to_vector,
)
from train_dqn import opponent_action, save_run_result, set_seed, summarize_result


def make_high_dim_opponent_policies():
    return {
        0: BaseStockPolicy(target_inventory=100, low=0, high=max(ORDER_LEVELS)),
        2: BaseStockPolicy(target_inventory=100, low=0, high=max(ORDER_LEVELS)),
    }


def build_high_dim_actions(env, agent, state, epsilon, opponent_policies):
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
            order = random.choice(ORDER_LEVELS)
            actions[firm_id] = scalar_order_to_vector(order)

    return actions, learning_action


def train_high_dim_dqn_agent(
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
            actions, learning_action = build_high_dim_actions(env, agent, state, eps, opponent_policies)
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


def test_high_dim_agent(env, agent, num_episodes=TEST_EPISODES, opponent_policies=None):
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
            actions, _learning_action = build_high_dim_actions(env, agent, state, 0.0, opponent_policies)
            next_state, rewards, done = env.step(actions)

            episode_inventory.append(env.inventory[agent.firm_id][0])
            episode_orders.append(env.orders[agent.firm_id][0])
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


def run_high_dim_dqn_experiment(
    name="high_dim_dqn",
    agent_class=DQNAgent,
    algorithm="high_dim_dqn",
    seed_offset=0,
    result_root="results",
    num_episodes=NUM_EPISODES,
    test_episodes=TEST_EPISODES,
):
    set_seed(SEED + seed_offset)
    env = make_env()
    opponent_policies = make_high_dim_opponent_policies()
    result_dir = Path(result_root) / name
    checkpoint_dir = result_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    agent = agent_class(
        state_size=3,
        action_size=ACTION_SIZE,
        firm_id=FIRM_ID,
        max_order=ACTION_SIZE,
    )
    training_scores = train_high_dim_dqn_agent(
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
    ) = test_high_dim_agent(
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
            "order_dim": ORDER_DIM,
            "action_size": ACTION_SIZE,
            "num_episodes": num_episodes,
            "test_episodes": test_episodes,
            "seed": SEED + seed_offset,
        },
    )
    return summary


if __name__ == "__main__":
    summary = run_high_dim_dqn_experiment()
    print("high_dim_dqn summary:", summary)
