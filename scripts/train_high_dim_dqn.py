"""Train and evaluate the high-dimensional DQN experiment."""

from pathlib import Path
from itertools import product
import random
import sys

import hydra
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beergame.policy.base_stock import BaseStockPolicy
from beergame.policy.dqn import DQNAgent
from beergame.policy.high_dim import (
    ENV_CONFIG,
    FIRM_ID,
    NUM_EPISODES,
    ORDER_LEVELS,
    SEED,
    TEST_EPISODES,
)
from beergame.sim.env import Env
from train_dqn import opponent_action, save_run_result, set_seed, summarize_result
from train_config import (
    as_plain_config,
    default_config_dict,
    hydra_config_to_dict,
    result_dir_for_run,
    runtime_finish,
    runtime_start,
)


EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995


def make_high_dim_opponent_policies(high_dim_config=None):
    order_levels = high_dim_config["order_levels"] if high_dim_config else ORDER_LEVELS
    return {
        0: BaseStockPolicy(target_inventory=100, low=0, high=max(order_levels)),
        2: BaseStockPolicy(target_inventory=100, low=0, high=max(order_levels)),
    }


def make_high_dim_env(env_config, high_dim_config):
    return Env(
        **env_config,
        order_dim=high_dim_config["order_dim"],
        order_cost_multipliers=high_dim_config["order_cost_multipliers"],
    )


def make_action_space(high_dim_config):
    return [
        action
        for action in product(high_dim_config["order_levels"], repeat=high_dim_config["order_dim"])
        if high_dim_config["min_total_order"] <= sum(action) <= high_dim_config["max_total_order"]
    ]


def action_to_order_vector(action, action_space):
    return np.asarray(action_space[int(action) - 1], dtype=float)


def scalar_order_to_vector(order, order_dim):
    order_vector = np.zeros(order_dim, dtype=float)
    order_vector[0] = order
    return order_vector


def build_high_dim_actions(env, agent, state, epsilon, opponent_policies, action_space, high_dim_config):
    order_dim = high_dim_config["order_dim"]
    actions = np.zeros((env.num_firms, order_dim))
    learning_action = None

    for firm_id in range(env.num_firms):
        firm_state = state[firm_id].reshape(1, -1)
        if firm_id == agent.firm_id:
            learning_action = agent.act(firm_state, epsilon)
            actions[firm_id] = action_to_order_vector(learning_action, action_space)
        elif opponent_policies and firm_id in opponent_policies:
            order = opponent_action(opponent_policies[firm_id], firm_state, env, firm_id)
            actions[firm_id] = scalar_order_to_vector(order, order_dim)
        else:
            order = random.choice(high_dim_config["order_levels"])
            actions[firm_id] = scalar_order_to_vector(order, order_dim)

    return actions, learning_action


def train_high_dim_dqn_agent(
    env,
    agent,
    num_episodes=NUM_EPISODES,
    max_t=ENV_CONFIG["max_steps"],
    eps_start=EPS_START,
    eps_end=EPS_END,
    eps_decay=EPS_DECAY,
    opponent_policies=None,
    checkpoint_prefix=None,
    final_model_path=None,
    action_space=None,
    high_dim_config=None,
):
    scores = []
    eps = eps_start

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        score = 0

        for _t in range(max_t):
            actions, learning_action = build_high_dim_actions(
                env,
                agent,
                state,
                eps,
                opponent_policies,
                action_space,
                high_dim_config,
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


def test_high_dim_agent(
    env,
    agent,
    num_episodes=TEST_EPISODES,
    opponent_policies=None,
    action_space=None,
    high_dim_config=None,
):
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
            actions, _learning_action = build_high_dim_actions(
                env,
                agent,
                state,
                0.0,
                opponent_policies,
                action_space,
                high_dim_config,
            )
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
    config=None,
    runtime=None,
):
    config = config or default_config_dict()
    training_config = config["training"]
    dqn_config = config["dqn"]
    env_config = config["env"]
    high_dim_config = config["high_dim"]
    action_space = make_action_space(high_dim_config)
    action_size = len(action_space)

    name = training_config.get("name", name)
    algorithm = training_config.get("algorithm", algorithm)
    result_root = training_config.get("result_root", result_root)
    seed_offset = training_config.get("seed_offset", seed_offset)
    num_episodes = training_config.get("num_episodes", num_episodes)
    test_episodes = training_config.get("test_episodes", test_episodes)
    firm_id = training_config.get("firm_id", FIRM_ID)
    seed = training_config.get("seed", SEED) + seed_offset
    eval_seed_offset = training_config.get("eval_seed_offset")

    set_seed(seed)
    env = make_high_dim_env(env_config, high_dim_config)
    opponent_policies = make_high_dim_opponent_policies(high_dim_config)
    result_dir = result_dir_for_run(config, name, result_root)
    checkpoint_dir = result_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    agent = agent_class(
        state_size=3,
        action_size=action_size,
        firm_id=firm_id,
        max_order=action_size,
        **dqn_config,
    )
    training_scores = train_high_dim_dqn_agent(
        env,
        agent,
        num_episodes=num_episodes,
        max_t=env_config["max_steps"],
        eps_start=training_config.get("eps_start", EPS_START),
        eps_end=training_config.get("eps_end", EPS_END),
        eps_decay=training_config.get("eps_decay", EPS_DECAY),
        opponent_policies=opponent_policies,
        checkpoint_prefix=str(checkpoint_dir / name),
        final_model_path=str(result_dir / "model.pth"),
        action_space=action_space,
        high_dim_config=high_dim_config,
    )
    if eval_seed_offset is not None:
        set_seed(seed + eval_seed_offset)
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
        action_space=action_space,
        high_dim_config=high_dim_config,
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
            **as_plain_config(config),
            "action_size": action_size,
            "effective_seed": seed,
        },
        runtime=runtime_finish(runtime) if runtime else None,
    )
    return summary


@hydra.main(version_base=None, config_path="../cfg", config_name="high_dim_dqn")
def main(cfg):
    cfg = hydra_config_to_dict(cfg)
    runtime = runtime_start()
    summary = run_high_dim_dqn_experiment(config=cfg, runtime=runtime)
    print("high_dim_dqn summary:", summary)


if __name__ == "__main__":
    main()
