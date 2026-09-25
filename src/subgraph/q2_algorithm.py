"""Question 2 algorithm entry point.

This module owns the current planning implementation.  The other question
modules deliberately import this entry point until their specialized
algorithms are implemented.
"""

from __future__ import annotations

from typing import Any

from .demo_algorithm import GraphFeatures, analyze_graph, build_algorithm_plan
from .interfaces import AlgorithmResult


def build_plan(
    graph: dict[str, Any],
    num_cores: int = 4,
    features: GraphFeatures | None = None,
) -> AlgorithmResult:
    """Build the current plan using the Q2 strategy."""
    if features is None:
        features = analyze_graph(graph)
    return build_algorithm_plan(features, num_cores, scenario="q2")

