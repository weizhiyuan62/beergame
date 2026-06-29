"""PPO agent and training utilities for the beer game."""

import os

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
