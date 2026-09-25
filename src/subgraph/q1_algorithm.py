"""Question 1: communication-aware cut and schedule.

In Question 1 every subgraph boundary is materialised, even when the two
subgraphs run on the same core.  This makes fixed boundary latency (100 cycles
on-core, 1000 cycles cross-core) part of the partitioning decision, not only a
scheduling detail.  The implementation reuses Q2's graph helpers but gives Q1
larger, boundary-averse blocks and a stronger critical-path affinity policy.
"""

from __future__ import annotations

from typing import Any

from .algorithm_common import GraphFeatures, Partition, _topological_order, analyze_graph
from .interfaces import AlgorithmResult
from .q2_algorithm import (
    _coalesce_topological_partitions,
    _critical_partition_path,
    _schedule_complex_partitions,
    _schedule_partitions,
)


def _q1_partitions(features: GraphFeatures, pattern: Any) -> tuple[list[Partition], dict[str, Any]]:
    """Build larger Q1 blocks while preserving parallelism in wide graphs."""
    from .graph_patterns import GraphPattern
    from .semantic_partition import semantic_partition

    depth = max(features.depth.values(), default=0)
    width = max(
        (sum(features.depth[node] == level for node in features.topo_order)
         for level in set(features.depth.values())),
        default=0,
    )
    if pattern in {GraphPattern.WIDE_MATMUL_ADD, GraphPattern.SHALLOW_WIDE_COMPUTE_ACTIVATION}:
        max_ops, max_cycles = (24, 32000) if width < 1000 else (20, 24000)
        if pattern == GraphPattern.WIDE_MATMUL_ADD and depth >= 16 and width < 1000:
            max_ops, max_cycles = 14, 14000
        motif = "q1_wide_boundary_aware"
    elif pattern == GraphPattern.CNN_RESIDUAL:
        max_ops, max_cycles = (28, 60000) if width > 16 else (24, 45000)
        motif = "q1_residual_fusion"
    elif pattern == GraphPattern.GATED_SIGMOID_MLP:
        max_ops, max_cycles = 28, 42000
        motif = "q1_gated_chain_fusion"
    else:
        max_ops, max_cycles = (22, 30000) if width >= 120 else (28, 42000)
        motif = "q1_semantic_fusion"

    partitions = semantic_partition(
        features,
        max_ops=max_ops,
        max_cycles=max_cycles,
        enable_singleton_repair=True,
    )
    # The semantic pass is intentionally cautious around joins.  Remove some
    # remaining serial boundaries with a bounded topological coalescing pass.
    if pattern != GraphPattern.CNN_RESIDUAL and len(partitions) > 1:
        partitions = _coalesce_topological_partitions(
            partitions, min(max_ops, 32), min(max_cycles, 50000)
        )
    return partitions, {
        "strategy": motif,
        "partition_count": len(partitions),
        "max_ops": max_ops,
        "max_cycles": max_cycles,
        "fixed_boundary_wait": 100,
        "cross_core_wait": 1000,
    }


def build_algorithm_plan(features: GraphFeatures, num_cores: int = 4) -> AlgorithmResult:
    """Build a Question 1 plan."""
    from .graph_patterns import GraphPatternFamily, classify_features

    if num_cores < 1:
        raise ValueError("num_cores must be positive")
    graph_pattern = classify_features(features)
    partitions, diagnostics = _q1_partitions(features, graph_pattern.pattern)
    critical_path = _critical_partition_path(partitions, features)

    if graph_pattern.family == GraphPatternFamily.COMPLEX and critical_path:
        core_orders = _schedule_complex_partitions(
            partitions, features, num_cores, "q1", critical_path
        )
        scheduler = "q1_critical_path_sticky"
    else:
        core_orders = _schedule_partitions(partitions, features, num_cores, "q1")
        scheduler = "q1_affinity_list"

    node_to_subgraph = {
        str(op_id): partition.id
        for partition in partitions
        for op_id in partition.ops
    }
    diagnostics.update({
        "scheduler": scheduler,
        "critical_path_length": len(critical_path),
        "graph_pattern": graph_pattern.as_dict(),
    })
    return AlgorithmResult(
        plan={"node_to_subgraph": node_to_subgraph, "core_schedules": core_orders},
        diagnostics={"algorithm": diagnostics},
    )


def build_plan(
    graph: dict[str, Any],
    num_cores: int = 4,
    features: GraphFeatures | None = None,
) -> AlgorithmResult:
    """Framework entry point for Question 1."""
    if features is None:
        features = analyze_graph(graph)
    return build_algorithm_plan(features, num_cores)


__all__ = ["build_plan", "build_algorithm_plan"]
