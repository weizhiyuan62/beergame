"""Train and evaluate the high-dimensional PPO experiment."""

from pathlib import Path
import random
import sys

import hydra
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beergame.policy.ppo import PPOAgent
from train_dqn import opponent_action, save_run_result, set_seed, summarize_result
from train_high_dim_dqn import (
    ENV_CONFIG,
    FIRM_ID,
    NUM_EPISODES,
    SEED,
    TEST_EPISODES,
    action_to_order_vector,
    make_action_space,
    make_high_dim_env,
    make_high_dim_opponent_policies,
    scalar_order_to_vector,
)
from train_ppo import REWARD_SCALE, RolloutBatch
from train_config import (
    as_plain_config,
    default_config_dict,
    hydra_config_to_dict,
    result_dir_for_run,
    runtime_finish,
    runtime_start,
)


def train_high_dim_ppo_agent(
    env,
    agent,
    num_episodes=NUM_EPISODES,
    max_t=ENV_CONFIG["max_steps"],
    opponent_policies=None,
    checkpoint_prefix=None,
    final_model_path=None,
    reward_scale=REWARD_SCALE,
    action_space=None,
    high_dim_config=None,
):
    """Train PPO over an enumerated high-dimensional order-vector action space."""
    scores = []
    order_dim = high_dim_config["order_dim"]

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        score = 0.0
        rollout_states = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_rewards = []
        rollout_dones = []
        rollout_values = []

        for _t in range(max_t):
            actions = np.zeros((env.num_firms, order_dim))
            learning_action = None
            log_prob = None
            value = None

            for firm_id in range(env.num_firms):
                firm_state = state[firm_id].reshape(1, -1)
                if firm_id == agent.firm_id:
                    learning_action, log_prob, value = agent.select_action(
                        firm_state,
                        deterministic=False,
                    )
                    actions[firm_id] = action_to_order_vector(learning_action, action_space)
                elif opponent_policies and firm_id in opponent_policies:
                    order = opponent_action(opponent_policies[firm_id], firm_state, env, firm_id)
                    actions[firm_id] = scalar_order_to_vector(order, order_dim)
                else:
                    order = random.choice(high_dim_config["order_levels"])
                    actions[firm_id] = scalar_order_to_vector(order, order_dim)

            next_state, rewards, done = env.step(actions)
            reward = float(rewards[agent.firm_id][0])

            rollout_states.append(state[agent.firm_id].reshape(-1))
            rollout_actions.append(int(learning_action) - 1)
            rollout_log_probs.append(log_prob)
            rollout_rewards.append(reward * reward_scale)
            rollout_dones.append(float(done))
            rollout_values.append(value)

            state = next_state
            score += reward

            if done:
                break

        if rollout_dones and rollout_dones[-1] == 1.0:
            next_value = 0.0
        else:
            _, _, next_value = agent.select_action(
                state[agent.firm_id].reshape(1, -1),
                deterministic=True,
            )

        returns, advantages = agent.compute_returns_and_advantages(
            rollout_rewards,
            rollout_dones,
            rollout_values,
            next_value,
        )

        batch = RolloutBatch(
            states=torch.as_tensor(np.asarray(rollout_states), dtype=torch.float32, device=agent.device),
            actions=torch.as_tensor(np.asarray(rollout_actions), dtype=torch.long, device=agent.device),
            log_probs=torch.as_tensor(np.asarray(rollout_log_probs), dtype=torch.float32, device=agent.device),
            returns=torch.as_tensor(returns, dtype=torch.float32, device=agent.device),
            advantages=torch.as_tensor(advantages, dtype=torch.float32, device=agent.device),
            values=torch.as_tensor(np.asarray(rollout_values), dtype=torch.float32, device=agent.device),
        )
        metrics = agent.update(batch)
        scores.append(score)

        if i_episode % 100 == 0:
            print(
                f"Episode {i_episode}/{num_episodes} | "
                f"Average Score: {np.mean(scores[-100:]):.2f} | "
                f"Policy Loss: {metrics['policy_loss']:.4f} | "
                f"Value Loss: {metrics['value_loss']:.4f}"
            )

        if i_episode % 500 == 0 and checkpoint_prefix:
            agent.save(f"{checkpoint_prefix}_episode_{i_episode}.pth")

    if final_model_path is not None:
        agent.save(final_model_path)

    return scores


def test_high_dim_ppo_agent(
    env,
    agent,
    num_episodes=TEST_EPISODES,
    opponent_policies=None,
    action_space=None,
    high_dim_config=None,
):
    """Evaluate a trained high-dimensional PPO agent deterministically."""
    scores = []
    inventory_history = []
    orders_history = []
    demand_history = []
    satisfied_demand_history = []
    order_dim = high_dim_config["order_dim"]

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        score = 0.0
        episode_inventory = []
        episode_orders = []
        episode_demand = []
        episode_satisfied_demand = []

        for _t in range(env.max_steps):
            actions = np.zeros((env.num_firms, order_dim))
            for firm_id in range(env.num_firms):
                firm_state = state[firm_id].reshape(1, -1)
                if firm_id == agent.firm_id:
                    action, _log_prob, _value = agent.select_action(
                        firm_state,
                        deterministic=True,
                    )
                    actions[firm_id] = action_to_order_vector(action, action_space)
                elif opponent_policies and firm_id in opponent_policies:
                    order = opponent_action(opponent_policies[firm_id], firm_state, env, firm_id)
                    actions[firm_id] = scalar_order_to_vector(order, order_dim)
                else:
                    order = random.choice(high_dim_config["order_levels"])
                    actions[firm_id] = scalar_order_to_vector(order, order_dim)

            next_state, rewards, done = env.step(actions)
            episode_inventory.append(env.inventory[agent.firm_id][0])
            episode_orders.append(env.orders[agent.firm_id][0])
            episode_demand.append(env.demand[agent.firm_id][0])
            episode_satisfied_demand.append(env.satisfied_demand[agent.firm_id][0])

            state = next_state
            score += float(rewards[agent.firm_id][0])

            if done:
                break

        scores.append(score)
        inventory_history.append(episode_inventory)
        orders_history.append(episode_orders)
        demand_history.append(episode_demand)
        satisfied_demand_history.append(episode_satisfied_demand)

        print(f"Test Episode {i_episode}/{num_episodes} | Score: {score:.2f}")

    return scores, inventory_history, orders_history, demand_history, satisfied_demand_history


def run_high_dim_ppo_experiment(
    name="high_dim_ppo",
    result_root="results",
    seed_offset=300,
    num_episodes=NUM_EPISODES,
    test_episodes=TEST_EPISODES,
    config=None,
    runtime=None,
):
    """Run one high-dimensional PPO experiment and save standard artifacts."""
    config = config or default_config_dict()
    training_config = config["training"]
    ppo_config = dict(config["ppo"])
    env_config = config["env"]
    high_dim_config = config["high_dim"]
    reward_scale = ppo_config.pop("reward_scale", REWARD_SCALE)
    action_space = make_action_space(high_dim_config)
    action_size = len(action_space)

    name = training_config.get("name", name)
    algorithm = training_config.get("algorithm", "high_dim_ppo")
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

    agent = PPOAgent(
        state_size=3,
        action_size=action_size,
        firm_id=firm_id,
        **ppo_config,
    )
    training_scores = train_high_dim_ppo_agent(
        env,
        agent,
        num_episodes=num_episodes,
        max_t=env_config["max_steps"],
        opponent_policies=opponent_policies,
        checkpoint_prefix=str(checkpoint_dir / name),
        final_model_path=str(result_dir / "model.pth"),
        reward_scale=reward_scale,
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
    ) = test_high_dim_ppo_agent(
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


@hydra.main(version_base=None, config_path="../cfg", config_name="high_dim_ppo")
def main(cfg):
    cfg = hydra_config_to_dict(cfg)
    runtime = runtime_start()
    summary = run_high_dim_ppo_experiment(config=cfg, runtime=runtime)
    print("high_dim_ppo summary:", summary)


if __name__ == "__main__":
    main()
