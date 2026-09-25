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


def _partition_edge_size(
    partitions: list[Partition],
    features: GraphFeatures,
    source: int,
    target: int,
) -> int:
    """Return the largest known tensor edge between two partitions."""
    source_ops = partitions[source].ops
    target_ops = partitions[target].ops
    return max(
        (
            _edge_size(features, source_op, target_op)
            for source_op in source_ops
            for target_op in target_ops
            if target_op in features.succs[source_op]
        ),
        default=0,
    )


def rebalance_core_orders(
    partitions: list[Partition],
    features: GraphFeatures,
    core_orders: list[list[int]],
    scenario: str,
    max_rounds: int = 3,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Improve a completed schedule with bounded whole-partition migration.

    The regular schedulers are greedy and make a local core decision for every
    ready partition.  This pass treats that result as a starting point and
    searches migrations from the heaviest core to the lightest cores.  A move
    is accepted only when the replayed dependency schedule improves a scalar
    estimate that combines makespan and a small communication term.  Orders
    are rebuilt in partition-topological order after every trial, so moving a
    partition cannot introduce a same-core dependency cycle.
    """
    if not core_orders or not partitions:
        return core_orders, {"enabled": True, "moves": 0}
    if scenario not in {"q1", "q2", "q3"}:
        raise ValueError("scenario must be q1, q2, or q3")

    by_id = {partition.id: partition for partition in partitions}
    topo = _topological_order(
        by_id,
        {partition.id: set(partition.preds) for partition in partitions},
        {partition.id: set(partition.succs) for partition in partitions},
    )
    topo_position = {pid: index for index, pid in enumerate(topo)}
    owner = {
        pid: core
        for core, order in enumerate(core_orders)
        for pid in order
    }
    # Defensive completion for malformed/empty schedules; normal plans already
    # contain every partition exactly once.
    for pid in topo:
        owner.setdefault(pid, min(range(len(core_orders)), key=lambda c: c))

    def rebuild(current_owner: dict[int, int]) -> list[list[int]]:
        orders = [[] for _ in core_orders]
        for pid in topo:
            orders[current_owner[pid]].append(pid)
        return orders

    def replay(current_owner: dict[int, int]) -> tuple[float, int, int, float]:
        orders = rebuild(current_owner)
        previous: dict[int, int] = {}
        finish: dict[int, int] = {}
        edge_bytes = 0
        for core, order in enumerate(orders):
            for index, pid in enumerate(order):
                previous[pid] = order[index - 1] if index else -1
        for pid in topo:
            core = current_owner[pid]
            ready = 0
            for pred in by_id[pid].preds:
                pred_core = current_owner[pred]
                cross = pred_core != core
                bytes_ = _partition_edge_size(partitions, features, pred, pid)
                if cross:
                    edge_bytes += bytes_
                if scenario == "q1":
                    delay = (100 if not cross else 1000) + math.ceil(bytes_ / 60)
                else:
                    delay = (500 + math.ceil(bytes_ / 60)) if cross else 0
                ready = max(ready, finish[pred] + delay)
            previous_finish = finish.get(previous[pid], 0)
            # Scene A releases the next Task on a core only after the
            # configured same-core Task wait.  This is a queueing constraint,
            # not a data-edge cost, so it applies even to unrelated Tasks.
            if scenario == "q1" and previous[pid] != -1:
                previous_finish += 100
            finish[pid] = max(ready, previous_finish) + by_id[pid].cycles
        loads = [
            sum(by_id[pid].cycles for pid in order)
            for order in orders
        ]
        makespan = max(finish.values(), default=0)
        max_load = max(loads, default=0)
        # Communication is a secondary term: compute/dependency time remains
        # dominant, while large migrations are discouraged when equal in time.
        score = makespan + 0.10 * edge_bytes / 60.0
        return score, edge_bytes, max_load, makespan

    initial = replay(owner)
    current = initial
    moves = 0
    rounds = 0
    candidate_limit = 24
    while rounds < max_rounds:
        rounds += 1
        orders = rebuild(owner)
        loads = [
            sum(by_id[pid].cycles for pid in order)
            for order in orders
        ]
        source = max(range(len(orders)), key=lambda core: (loads[core], -core))
        targets = sorted(
            (core for core in range(len(orders)) if core != source),
            key=lambda core: (loads[core], core),
        )
        if not targets or loads[source] <= loads[targets[0]]:
            break
        candidates = sorted(
            orders[source],
            key=lambda pid: (-by_id[pid].cycles, topo_position[pid]),
        )[:candidate_limit]
        best_owner = None
        best = current
        for pid in candidates:
            for target in targets:
                trial_owner = dict(owner)
                trial_owner[pid] = target
                trial = replay(trial_owner)
                if trial[0] + 1e-9 < best[0]:
                    best = trial
                    best_owner = trial_owner
        if best_owner is None:
            break
        owner = best_owner
        current = best
        moves += 1

    result = rebuild(owner)
    final = replay(owner)
    return result, {
        "enabled": True,
        "moves": moves,
        "rounds": rounds,
        "estimated_before": initial[3],
        "estimated_after": final[3],
        "estimated_cross_bytes": final[1],
    }


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
    "rebalance_core_orders",
]
