"""Base-stock order policy for controlled supply-chain comparisons."""

import numpy as np


class BaseStockPolicy:
    def __init__(self, target_inventory=100, low=1, high=20):
        """
        Initialize a base-stock policy.

        :param target_inventory: Desired inventory position.
        :param low: Inclusive lower bound for order quantity.
        :param high: Inclusive upper bound for order quantity.
        """
        self.target_inventory = target_inventory
        self.low = low
        self.high = high

    def act(self, state=None, *_args, **_kwargs):
        """Order enough to move current inventory toward the target level."""
        if state is None:
            return self.low

        firm_state = np.asarray(state).flatten()
        inventory = firm_state[2]
        order = self.target_inventory - inventory
        return int(np.clip(order, self.low, self.high))
