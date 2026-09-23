"""Selectable ``module:callable`` entry point for the historical algorithm.

Use ``subgraph.naive_algorithm:build_plan`` with the evaluation tool.  The
returned object remains the official two-field plan, while
``include_diagnostics=True`` exposes the optional scheduler diagnostics for
experiments.
"""

from __future__ import annotations

from typing import Any

from .demo_framework import GraphFeatures, analyze_graph, schedule_partitions
from .naive_partition import naive_partition


def build_plan(
    graph: dict[str, Any],
    num_cores: int = 4,
    scenario: str = "q2",
    include_diagnostics: bool = False,
) -> dict[str, Any]:
    """Build a standard plan using the historical greedy partitioner."""
    features: GraphFeatures = analyze_graph(graph)
    partitions = naive_partition(features)
    scheduled = schedule_partitions(
        partitions,
        features,
        num_cores,
        scenario,
        return_diagnostics=include_diagnostics,
    )
    if include_diagnostics:
        core_of, core_orders, diagnostics = scheduled
    else:
        core_of, core_orders = scheduled

    plan: dict[str, Any] = {
        "node_to_subgraph": {
            str(op_id): partition.id
            for partition in partitions
            for op_id in partition.ops
        },
        "core_schedules": core_orders,
    }
    if include_diagnostics:
        plan["diagnostics"] = diagnostics
    return plan


__all__ = ["build_plan"]
