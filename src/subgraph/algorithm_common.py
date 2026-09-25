"""Shared graph-analysis types and helpers for question algorithms.

The framework deliberately stops at the contest output boundary.  It does not
create COPY operations or simulate the NPU; the official evaluators do that.
The implementation is intentionally plain:

* Kahn topological sort for graph analysis;
* critical-path-first list scheduling on identical cores.

Graph-pattern strategies own both partitioning and scheduling.  The current
strategies all share the semantic partitioner plus the same list scheduler,
but the framework treats each pattern branch as one complete algorithm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from .interfaces import AlgorithmResult


COPY_TYPES = {"COPY_IN", "COPY_OUT"}
ANCHOR_TYPES = {"MATMUL", "CONV", "REDUCE"}
ELEMENTWISE_TYPES = {
    "ADD", "MUL", "SUB", "DIV", "RELU", "SIGMOID", "EXP", "SQRT", "NEG"
}


@dataclass
class GraphFeatures:
    """The small set of features consumed by the baseline algorithm."""

    op_by_id: dict[int, dict[str, Any]]
    tensor_by_id: dict[int, dict[str, Any]]
    preds: dict[int, set[int]]
    succs: dict[int, set[int]]
    edge_sizes: dict[tuple[int, int], int]
    topo_order: list[int]
    depth: dict[int, int]
    rank_u: dict[int, int]
    semantic: dict[int, str]
    input_tensors: dict[int, set[int]]
    output_tensors: dict[int, set[int]]


@dataclass
class Partition:
    id: int
    ops: list[int]
    cycles: int
    rank_u: int
    pipe_cycles: dict[str, int] = field(default_factory=dict)
    preds: set[int] = field(default_factory=set)
    succs: set[int] = field(default_factory=set)


def classify_op(op_type: str) -> str:
    if op_type in {"MATMUL", "CONV"}:
        return "DENSE_COMPUTE"
    if op_type == "REDUCE":
        return "REDUCTION"
    if op_type in ELEMENTWISE_TYPES:
        return "ELEMENTWISE"
    if op_type.startswith("COPY"):
        return "COMMUNICATION"
    return "OTHER"


def _op_graph(graph: dict[str, Any]) -> tuple[dict[int, set[int]], dict[int, set[int]], dict[tuple[int, int], int]]:
    """Convert op->tensor->op and direct op->op edges into an op DAG."""
    op_ids = {op["id"] for op in graph.get("ops", [])}
    tensor_by_id = {tensor["id"]: tensor for tensor in graph.get("tensors", [])}
    preds = {op_id: set() for op_id in op_ids}
    succs = {op_id: set() for op_id in op_ids}
    producers: dict[int, set[int]] = {}
    consumers: dict[int, set[int]] = {}
    edge_sizes: dict[tuple[int, int], int] = {}

    for edge in graph.get("edges", []):
        source, target = edge["source"], edge["target"]
        if source in op_ids and target in op_ids:
            if source != target:
                succs[source].add(target)
                preds[target].add(source)
                edge_sizes.setdefault((source, target), 0)
        elif source in op_ids and target in tensor_by_id:
            producers.setdefault(target, set()).add(source)
        elif source in tensor_by_id and target in op_ids:
            consumers.setdefault(source, set()).add(target)

    for tensor_id, source_ids in producers.items():
        size = int(tensor_by_id[tensor_id].get("size", 0))
        for source in source_ids:
            for target in consumers.get(tensor_id, set()):
                if source == target:
                    continue
                succs[source].add(target)
                preds[target].add(source)
                # A producer/consumer pair may be connected by multiple
                # tensors.  Communication is the sum of those tensor bytes,
                # not the largest tensor only.
                edge_sizes[(source, target)] = edge_sizes.get((source, target), 0) + size
    return preds, succs, edge_sizes


def _topological_order(nodes: Iterable[int], preds: dict[int, set[int]], succs: dict[int, set[int]]) -> list[int]:
    indegree = {node: len(preds[node]) for node in nodes}
    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    result: list[int] = []
    while ready:
        node = ready.pop(0)
        result.append(node)
        for successor in sorted(succs[node]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(result) != len(indegree):
        raise ValueError("the operation graph contains a cycle")
    return result


def analyze_graph(graph: dict[str, Any]) -> GraphFeatures:
    op_by_id = {op["id"]: op for op in graph.get("ops", [])}
    tensor_by_id = {tensor["id"]: tensor for tensor in graph.get("tensors", [])}
    preds, succs, edge_sizes = _op_graph(graph)
    producers: dict[int, set[int]] = {}
    consumers: dict[int, set[int]] = {}
    op_ids = set(op_by_id)
    for edge in graph.get("edges", []):
        source, target = edge["source"], edge["target"]
        if source in op_ids and target in tensor_by_id:
            producers.setdefault(target, set()).add(source)
        elif source in tensor_by_id and target in op_ids:
            consumers.setdefault(source, set()).add(target)
    eligible = [op_id for op_id, op in op_by_id.items() if op.get("op") not in COPY_TYPES]

    # COPY nodes are boundaries, not user partitions.  Contract them by walking
    # through COPY nodes before computing the eligible-op topology.
    contracted_preds = {node: set() for node in eligible}
    contracted_succs = {node: set() for node in eligible}
    eligible_set = set(eligible)
    contracted_edge_sizes: dict[tuple[int, int], int] = {}
    for source in eligible:
        # Keep the largest boundary tensor-path estimate seen for each COPY
        # node.  COPY contraction can expose several paths, but revisiting a
        # node is only useful when the carried byte estimate increases.
        stack = [(target, edge_sizes.get((source, target), 0)) for target in succs[source]]
        seen_weight: dict[int, int] = {}
        while stack:
            target, path_bytes = stack.pop()
            if target in eligible_set:
                if target != source:
                    contracted_succs[source].add(target)
                    contracted_preds[target].add(source)
                    key = (source, target)
                    contracted_edge_sizes[key] = max(
                        contracted_edge_sizes.get(key, 0), path_bytes
                    )
                continue
            if path_bytes <= seen_weight.get(target, -1):
                continue
            seen_weight[target] = path_bytes
            stack.extend(
                (next_target, max(path_bytes, edge_sizes.get((target, next_target), 0)))
                for next_target in succs[target]
            )

    topo = _topological_order(eligible, contracted_preds, contracted_succs)
    depth: dict[int, int] = {}
    rank_u: dict[int, int] = {}
    for node in topo:
        depth[node] = 0 if not contracted_preds[node] else 1 + max(depth[p] for p in contracted_preds[node])
    for node in reversed(topo):
        cycles = int(op_by_id[node].get("cycles", 0))
        rank_u[node] = cycles if not contracted_succs[node] else cycles + max(
            rank_u[s] for s in contracted_succs[node]
        )

    # Return contracted dependencies for non-COPY operations.  The raw edge-size
    # lookup remains useful for direct edges; tensor edges are recovered below.
    input_tensors = {node: set() for node in eligible}
    output_tensors = {node: set() for node in eligible}
    for tensor_id, source_ids in producers.items():
        for source in source_ids & eligible_set:
            output_tensors[source].add(tensor_id)
            for target in consumers.get(tensor_id, set()) & eligible_set:
                input_tensors[target].add(tensor_id)
    return GraphFeatures(
        op_by_id=op_by_id,
        tensor_by_id=tensor_by_id,
        preds=contracted_preds,
        succs=contracted_succs,
        edge_sizes=contracted_edge_sizes,
        topo_order=topo,
        depth=depth,
        rank_u=rank_u,
        semantic={node: classify_op(op_by_id[node].get("op", "")) for node in eligible},
        input_tensors=input_tensors,
        output_tensors=output_tensors,
    )


def _edge_size(features: GraphFeatures, source: int, target: int) -> int:
    # Missing edge metadata means there is no known tensor transfer.  Falling
    # back to the largest tensor in the whole graph severely overestimates
    # unrelated dependencies after COPY contraction.
    return features.edge_sizes.get((source, target), 0)


def describe_graph_pattern(features: GraphFeatures) -> dict[str, Any]:
    """Return the graph-pattern diagnostics emitted with each plan."""
    from .graph_patterns import classify_features

    return classify_features(features).as_dict()



__all__ = [
    "GraphFeatures",
    "Partition",
    "analyze_graph",
    "classify_op",
    "describe_graph_pattern",
]
