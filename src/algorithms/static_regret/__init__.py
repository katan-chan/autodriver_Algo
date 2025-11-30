"""Static (non-time-aware) greedy-regret routing algorithms."""

from .baseline import solve_routing_without_penalty
from .greedy_regret import solve_routing_with_penalty_greedy_regret

__all__ = [
    "solve_routing_without_penalty",
    "solve_routing_with_penalty_greedy_regret",
]
