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
        order_dim=1,
        order_cost_multipliers=None,
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
        :param order_dim: Number of order channels in each firm's action.
        :param order_cost_multipliers: Cost multiplier for each order channel.
        """
        self.num_firms = num_firms
        self.p = p
        self.h = h
        self.c = c
        self.poisson_lambda = poisson_lambda
        self.max_steps = max_steps
        self.initial_inventory = initial_inventory
        self.order_dim = order_dim
        if order_cost_multipliers is None:
            self.order_cost_multipliers = np.ones(order_dim)
        else:
            self.order_cost_multipliers = np.asarray(order_cost_multipliers, dtype=float)
            if self.order_cost_multipliers.shape != (order_dim,):
                raise ValueError("order_cost_multipliers must have length order_dim")

        self.inventory = np.full((num_firms, 1), initial_inventory)
        self.order_vectors = np.zeros((num_firms, order_dim))
        self.orders = np.zeros((num_firms, 1))
        self.satisfied_demand = np.zeros((num_firms, 1))
        self.current_step = 0
        self.done = False

    def reset(self):
        """Reset the environment state and return the initial observation."""
        self.inventory = np.full((self.num_firms, 1), self.initial_inventory)
        self.order_vectors = np.zeros((self.num_firms, self.order_dim))
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

    def _normalize_actions(self, actions):
        """Convert scalar or vector actions to order vectors and total orders."""
        order_vectors = np.asarray(actions, dtype=float)
        if self.order_dim == 1:
            order_vectors = order_vectors.reshape(self.num_firms, 1)
        else:
            expected_shape = (self.num_firms, self.order_dim)
            if order_vectors.shape != expected_shape:
                raise ValueError(f"actions must have shape {expected_shape}")
        return order_vectors, np.sum(order_vectors, axis=1, keepdims=True)

    def step(self, actions):
        """
        Execute one simulation step.

        :param actions: Order quantity for each firm. Shape is
            ``(num_firms, 1)`` for the original scalar action space and
            ``(num_firms, order_dim)`` for high-dimensional order vectors.
        :return: ``next_state, rewards, done``.
        """
        self.order_vectors, self.orders = self._normalize_actions(actions)
        self.demand = self._generate_demand()

        for i in range(self.num_firms):
            self.satisfied_demand[i] = min(self.demand[i], self.inventory[i])

        for i in range(self.num_firms):
            self.inventory[i] = self.inventory[i] + self.orders[i] - self.satisfied_demand[i]

        rewards = np.zeros((self.num_firms, 1))
        loss_sales = np.zeros((self.num_firms, 1))

        for i in range(self.num_firms):
            upstream_price = self.p[i + 1] if i + 1 < self.num_firms else 0
            order_cost = upstream_price * np.dot(
                self.order_vectors[i],
                self.order_cost_multipliers,
            )
            rewards[i] += (
                self.p[i] * self.satisfied_demand[i]
                - order_cost
                - self.h * self.inventory[i]
            )

            if self.satisfied_demand[i] < self.demand[i]:
                loss_sales[i] = (self.demand[i] - self.satisfied_demand[i]) * self.c

        rewards -= loss_sales
        self.current_step += 1

        if self.current_step >= self.max_steps:
            self.done = True

        return self._get_observation(), rewards, self.done
