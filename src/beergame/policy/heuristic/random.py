"""Random order policy used as a simple baseline for non-learning firms."""

import numpy as np


class RandomOrderPolicy:
    def __init__(self, low=1, high=20):
        """
        Initialize a random integer order policy.

        :param low: Inclusive lower bound for order quantity.
        :param high: Inclusive upper bound for order quantity.
        """
        self.low = low
        self.high = high

    def act(self, *_args, **_kwargs):
        """Return one random order quantity."""
        return np.random.randint(self.low, self.high + 1)
