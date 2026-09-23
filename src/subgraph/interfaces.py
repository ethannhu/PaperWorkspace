"""Small callable interfaces used by the planning pipeline.

Algorithms are ordinary callables on purpose.  A new experiment only needs to
implement one function and can be passed to :func:`build_plan` directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .demo_framework import GraphFeatures, Partition


@dataclass(frozen=True)
class AlgorithmResult:
    """The algorithm result passed from an algorithm to the evaluator."""

    plan: dict[str, Any]
    diagnostics: dict[str, Any]


Partitioner = Callable[["GraphFeatures", int, int], list["Partition"]]
# The scheduler and optimizer accept a small set of keyword options (for
# example ``return_diagnostics`` and ``max_iterations``), so their callable
# signatures are intentionally open rather than hidden behind base classes.
Scheduler = Callable[..., tuple[dict[int, int], list[list[int]]]]
ScheduleOptimizer = Callable[..., tuple[dict[int, int], list[list[int]], dict[str, Any]]]


__all__ = ["AlgorithmResult", "Partitioner", "Scheduler", "ScheduleOptimizer"]
