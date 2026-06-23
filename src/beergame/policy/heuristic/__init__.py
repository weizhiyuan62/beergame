"""Heuristic and management-style supply-chain policies."""

from beergame.policy.heuristic.base_stock import BaseStockPolicy
from beergame.policy.heuristic.random import RandomOrderPolicy

__all__ = ["BaseStockPolicy", "RandomOrderPolicy"]
