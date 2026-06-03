"""Basic beer game supply chain simulation environment."""

import numpy as np


class Env:
    def __init__(
        self,
        num_firms,
        p,
        h,
        c,
        initial_inventory,
        poisson_lambda=10,
        max_steps=100,
    ):
        """
        Initialize the supply chain simulation environment.

        :param num_firms: Number of firms.
        :param p: Price list for each firm.
        :param h: Inventory holding cost.
        :param c: Lost sales cost.
        :param initial_inventory: Initial inventory for each firm.
        :param poisson_lambda: Mean of downstream external demand.
        :param max_steps: Maximum steps in each episode.
        """
        self.num_firms = num_firms
        self.p = p
        self.h = h
        self.c = c
        self.poisson_lambda = poisson_lambda
        self.max_steps = max_steps
        self.initial_inventory = initial_inventory

        self.inventory = np.full((num_firms, 1), initial_inventory)
        self.orders = np.zeros((num_firms, 1))
        self.satisfied_demand = np.zeros((num_firms, 1))
        self.current_step = 0
        self.done = False

    def reset(self):
        """Reset the environment state and return the initial observation."""
        self.inventory = np.full((self.num_firms, 1), self.initial_inventory)
        self.orders = np.zeros((self.num_firms, 1))
        self.satisfied_demand = np.zeros((self.num_firms, 1))
        self.current_step = 0
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        """
        Return each firm's local observation: orders, satisfied demand, inventory.
        """
        return np.concatenate((self.orders, self.satisfied_demand, self.inventory), axis=1)

    def _generate_demand(self):
        """
        Generate demand for each firm.

        The downstream firm faces Poisson external demand. Other firms receive
        demand equal to the downstream firm's order.
        """
        demand = np.zeros((self.num_firms, 1))
        for i in range(self.num_firms):
            if i == 0:
                demand[i] = np.random.poisson(self.poisson_lambda)
            else:
                demand[i] = self.orders[i - 1]
        return demand

    def step(self, actions):
        """
        Execute one simulation step.

        :param actions: Order quantity for each firm, shape ``(num_firms, 1)``.
        :return: ``next_state, rewards, done``.
        """
        self.orders = actions
        self.demand = self._generate_demand()

        for i in range(self.num_firms):
            self.satisfied_demand[i] = min(self.demand[i], self.inventory[i])

        for i in range(self.num_firms):
            self.inventory[i] = self.inventory[i] + self.orders[i] - self.satisfied_demand[i]

        rewards = np.zeros((self.num_firms, 1))
        loss_sales = np.zeros((self.num_firms, 1))

        for i in range(self.num_firms):
            upstream_price = self.p[i + 1] if i + 1 < self.num_firms else 0
            rewards[i] += (
                self.p[i] * self.satisfied_demand[i]
                - upstream_price * self.orders[i]
                - self.h * self.inventory[i]
            )

            if self.satisfied_demand[i] < self.demand[i]:
                loss_sales[i] = (self.demand[i] - self.satisfied_demand[i]) * self.c

        rewards -= loss_sales
        self.current_step += 1

        if self.current_step >= self.max_steps:
            self.done = True

        return self._get_observation(), rewards, self.done
