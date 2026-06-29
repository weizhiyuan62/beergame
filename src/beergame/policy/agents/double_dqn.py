"""Double DQN agent for the beer game baseline environment."""

import numpy as np
import torch
import torch.nn as nn

from beergame.policy.agents.dqn import DQNAgent


class DoubleDQNAgent(DQNAgent):
    """DQN variant that decouples next-action selection and evaluation."""

    def learn(self, experiences):
        """
        Learn from a batch of experiences using the Double DQN target.

        The online network chooses the next action, while the target network
        evaluates that chosen action. This reduces standard DQN's max-target
        overestimation bias.
        """
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.from_numpy(np.vstack([s.flatten() for s in states])).float()
        actions = torch.from_numpy(np.vstack([a - 1 for a in actions])).long()
        rewards = torch.from_numpy(np.vstack(rewards)).float()
        next_states = torch.from_numpy(np.vstack([ns.flatten() for ns in next_states])).float()
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()

        next_actions = self.q_network(next_states).detach().argmax(1).unsqueeze(1)
        Q_targets_next = self.target_network(next_states).detach().gather(1, next_actions)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_expected = self.q_network(states).gather(1, actions)
        loss = nn.MSELoss()(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learning_step += 1
        if self.learning_step % self.update_every == 0:
            self.soft_update()

        return loss.item()
