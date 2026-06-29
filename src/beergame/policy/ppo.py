"""PPO agent and training utilities for the beer game."""

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCriticNetwork(nn.Module):
    """Shared-backbone actor-critic for discrete order decisions."""

    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, state):
        features = self.backbone(state)
        return self.policy_head(features), self.value_head(features)


@dataclass
class RolloutBatch:
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor


class PPOAgent:
    """Single-agent PPO for the scalar Beer Game action space."""

    def __init__(
        self,
        state_size,
        action_size,
        firm_id,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        update_epochs=10,
        minibatch_size=32,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        device=None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.firm_id = firm_id
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = device or torch.device("cpu")

        self.network = ActorCriticNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

    def _state_tensor(self, state):
        return torch.as_tensor(state.flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)

    def select_action(self, state, deterministic=False):
        """Return a 1-based action, its log-probability, and state-value estimate."""
        state_tensor = self._state_tensor(state)
        self.network.eval()
        with torch.no_grad():
            logits, value = self.network(state_tensor)
            dist = Categorical(logits=logits)
            if deterministic:
                action_idx = torch.argmax(logits, dim=1)
            else:
                action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
        self.network.train()
        return (
            int(action_idx.item()) + 1,
            float(log_prob.item()),
            float(value.item()),
        )

    def act(self, state, epsilon=0.0):
        del epsilon
        action, _log_prob, _value = self.select_action(state, deterministic=True)
        return action

    def evaluate_actions(self, states, actions):
        logits, values = self.network(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values.squeeze(-1)

    def compute_returns_and_advantages(self, rewards, dones, values, next_value):
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        values = np.asarray(values + [next_value], dtype=np.float32)

        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1.0 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[step]) * gae
            advantages[step] = gae

        returns = advantages + values[:-1]
        return returns, advantages

    def update(self, batch):
        if len(batch.states) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        advantages = batch.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        dataset_size = batch.states.size(0)
        last_metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        for _ in range(self.update_epochs):
            indices = torch.randperm(dataset_size, device=self.device)
            for start in range(0, dataset_size, self.minibatch_size):
                mb_idx = indices[start:start + self.minibatch_size]
                states = batch.states[mb_idx]
                actions = batch.actions[mb_idx]
                old_log_probs = batch.log_probs[mb_idx]
                returns = batch.returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                new_log_probs, entropy, values = self.evaluate_actions(states, actions)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surrogate_1 = ratio * mb_advantages
                surrogate_2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                ) * mb_advantages

                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
                value_loss = nn.MSELoss()(values, returns)
                entropy_bonus = entropy.mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                last_metrics = {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy_bonus.item()),
                }

        return last_metrics

    def save(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filename,
        )
        print(f"模型已保存到 {filename}")

    def load(self, filename):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location=self.device)
            self.network.load_state_dict(checkpoint["network_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"从 {filename} 加载了模型")
            return True
        return False


def _opponent_action(policy, state, env, firm_id):
    try:
        return policy.act(state, env=env, firm_id=firm_id)
    except TypeError:
        try:
            return policy.act(state)
        except TypeError:
            return policy.act()


def train_ppo(
    env,
    agent,
    num_episodes=500,
    max_t=100,
    opponent_policies=None,
    checkpoint_prefix=None,
    final_model_path=None,
):
    """Train PPO with one on-policy rollout per episode."""
    scores = []

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
            actions = np.zeros((env.num_firms, 1))

            for firm_id in range(env.num_firms):
                firm_state = state[firm_id].reshape(1, -1)
                if firm_id == agent.firm_id:
                    action, log_prob, value = agent.select_action(firm_state, deterministic=False)
                    actions[firm_id] = action
                elif opponent_policies and firm_id in opponent_policies:
                    actions[firm_id] = _opponent_action(
                        opponent_policies[firm_id],
                        firm_state,
                        env,
                        firm_id,
                    )
                else:
                    actions[firm_id] = random.randint(1, agent.action_size)

            next_state, rewards, done = env.step(actions)
            reward = float(rewards[agent.firm_id][0])

            rollout_states.append(state[agent.firm_id].reshape(-1))
            rollout_actions.append(int(actions[agent.firm_id][0]) - 1)
            rollout_log_probs.append(log_prob)
            rollout_rewards.append(reward)
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

    if final_model_path is None:
        agent.save(f"models/ppo_agent_firm_{agent.firm_id}_final.pth")
    else:
        agent.save(final_model_path)

    return scores


def test_ppo_agent(env, agent, num_episodes=10, opponent_policies=None):
    """Evaluate PPO with greedy action selection."""
    scores = []
    inventory_history = []
    orders_history = []
    demand_history = []
    satisfied_demand_history = []

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        score = 0.0
        episode_inventory = []
        episode_orders = []
        episode_demand = []
        episode_satisfied_demand = []

        for _t in range(env.max_steps):
            actions = np.zeros((env.num_firms, 1))
            for firm_id in range(env.num_firms):
                firm_state = state[firm_id].reshape(1, -1)
                if firm_id == agent.firm_id:
                    action, _log_prob, _value = agent.select_action(firm_state, deterministic=True)
                    actions[firm_id] = action
                elif opponent_policies and firm_id in opponent_policies:
                    actions[firm_id] = _opponent_action(
                        opponent_policies[firm_id],
                        firm_state,
                        env,
                        firm_id,
                    )
                else:
                    actions[firm_id] = random.randint(1, agent.action_size)

            next_state, rewards, done = env.step(actions)
            reward = float(rewards[agent.firm_id][0])

            episode_inventory.append(env.inventory[agent.firm_id][0])
            episode_orders.append(actions[agent.firm_id][0])
            episode_demand.append(env.demand[agent.firm_id][0])
            episode_satisfied_demand.append(env.satisfied_demand[agent.firm_id][0])

            state = next_state
            score += reward

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
