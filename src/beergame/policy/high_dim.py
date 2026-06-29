"""High-dimensional order-space experiment for the beer game."""

from itertools import product

import numpy as np

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
ORDER_DIM = 3
ORDER_LEVELS = [0, 5, 10, 15, 20]
ORDER_COST_MULTIPLIERS = [1.0, 1.1, 1.25]
MIN_TOTAL_ORDER = 5
MAX_TOTAL_ORDER = 20
ACTION_SPACE = [
    action
    for action in product(ORDER_LEVELS, repeat=ORDER_DIM)
    if MIN_TOTAL_ORDER <= sum(action) <= MAX_TOTAL_ORDER
]
ACTION_SIZE = len(ACTION_SPACE)
NUM_EPISODES = 1500
TEST_EPISODES = 20
SEED = 42


def make_env():
    """Create the high-dimensional order-space environment."""
    return Env(
        **ENV_CONFIG,
        order_dim=ORDER_DIM,
        order_cost_multipliers=ORDER_COST_MULTIPLIERS,
    )


def action_to_order_vector(action):
    """Map the DQN's 1-based discrete action to a K-dimensional order vector."""
    return np.asarray(ACTION_SPACE[int(action) - 1], dtype=float)


def scalar_order_to_vector(order):
    """Place a scalar heuristic order in the lowest-cost channel."""
    order_vector = np.zeros(ORDER_DIM, dtype=float)
    order_vector[0] = order
    return order_vector
