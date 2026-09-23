"""The original conservative greedy partitioner.

This module preserves the partitioning implementation that preceded the
semantic partitioner.  It only fuses an elementwise operation into its
predecessor when the edge is a straight-line connection and the partition
limits are respected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .demo_framework import GraphFeatures, Partition


def naive_partition(
    features: GraphFeatures,
    max_ops: int = 16,
    max_cycles: int = 20000,
) -> list[Partition]:
    """Build partitions using the historical one-pass greedy rule."""
    if max_ops < 1 or max_cycles < 1:
        raise ValueError("partition limits must be positive")

    # Import lazily so importing the selectable module does not introduce a
    # cycle while demo_framework dispatches to this implementation.
    from .demo_framework import Partition

    partitions: list[Partition] = []
    op_to_partition: dict[int, int] = {}
    for node in features.topo_order:
        op = features.op_by_id[node]
        pred_ids = features.preds[node]
        candidate = None
        if len(pred_ids) == 1 and features.semantic[node] == "ELEMENTWISE":
            pred = next(iter(pred_ids))
            pred_partition = op_to_partition.get(pred)
            if pred_partition is not None and len(features.succs[pred]) == 1:
                partition = partitions[pred_partition]
                cycles = int(op.get("cycles", 0))
                if (
                    len(partition.ops) < max_ops
                    and partition.cycles + cycles <= max_cycles
                ):
                    candidate = partition
        if candidate is None:
            candidate = Partition(
                id=len(partitions),
                ops=[],
                cycles=0,
                rank_u=features.rank_u[node],
            )
            partitions.append(candidate)

        cycles = int(op.get("cycles", 0))
        candidate.ops.append(node)
        candidate.cycles += cycles
        pipe = str(op.get("pipe", "UNKNOWN"))
        candidate.pipe_cycles[pipe] = candidate.pipe_cycles.get(pipe, 0) + cycles
        candidate.rank_u = max(candidate.rank_u, features.rank_u[node])
        op_to_partition[node] = candidate.id

    for source in features.topo_order:
        for target in features.succs[source]:
            source_partition = op_to_partition[source]
            target_partition = op_to_partition[target]
            if source_partition != target_partition:
                partitions[source_partition].succs.add(target_partition)
                partitions[target_partition].preds.add(source_partition)
    return partitions


__all__ = ["naive_partition"]
