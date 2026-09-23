"""A small, deterministic graph partitioning and multicore scheduling framework.

The framework deliberately stops at the contest output boundary.  It does not
create COPY operations or simulate the NPU; the official evaluators do that.
The implementation is intentionally plain:

* Kahn topological sort for graph analysis;
* one-pass linear fusion of elementwise operators;
* critical-path-first list scheduling on identical cores.

This is a useful baseline for all three questions.  ``scenario`` only changes
the communication penalty used while assigning partitions to cores.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


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


@dataclass
class Partition:
    id: int
    ops: list[int]
    cycles: int
    rank_u: int
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
                edge_sizes[(source, target)] = 0
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
                edge_sizes[(source, target)] = max(edge_sizes.get((source, target), 0), size)
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
    eligible = [op_id for op_id, op in op_by_id.items() if op.get("op") not in COPY_TYPES]

    # COPY nodes are boundaries, not user partitions.  Contract them by walking
    # through COPY nodes before computing the eligible-op topology.
    contracted_preds = {node: set() for node in eligible}
    contracted_succs = {node: set() for node in eligible}
    eligible_set = set(eligible)
    for source in eligible:
        stack = list(succs[source])
        seen: set[int] = set()
        while stack:
            target = stack.pop()
            if target in eligible_set:
                if target != source:
                    contracted_succs[source].add(target)
                    contracted_preds[target].add(source)
                continue
            if target in seen:
                continue
            seen.add(target)
            stack.extend(succs[target])

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
    return GraphFeatures(
        op_by_id=op_by_id,
        tensor_by_id=tensor_by_id,
        preds=contracted_preds,
        succs=contracted_succs,
        edge_sizes=edge_sizes,
        topo_order=topo,
        depth=depth,
        rank_u=rank_u,
        semantic={node: classify_op(op_by_id[node].get("op", "")) for node in eligible},
    )


def _edge_size(features: GraphFeatures, source: int, target: int) -> int:
    if (source, target) in features.edge_sizes:
        return features.edge_sizes[(source, target)]
    # COPY contraction can hide the tensor edge.  A small conservative fallback
    # is enough for this baseline and keeps the partitioner independent of the
    # evaluator implementation.
    sizes = [int(t.get("size", 0)) for t in features.tensor_by_id.values()]
    return max(sizes, default=0)


def build_partitions(features: GraphFeatures, max_ops: int = 16, max_cycles: int = 20000) -> list[Partition]:
    """Fuse only a straight-line anchor -> elementwise suffix.

    A node is attached to its predecessor's partition only when it is the sole
    successor, has one predecessor, and does not create a large partition.  All
    other nodes start a new partition.  This is intentionally conservative and
    deterministic.
    """
    if max_ops < 1 or max_cycles < 1:
        raise ValueError("partition limits must be positive")
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
                p = partitions[pred_partition]
                if len(p.ops) < max_ops and p.cycles + int(op.get("cycles", 0)) <= max_cycles:
                    candidate = p
        if candidate is None:
            candidate = Partition(
                id=len(partitions), ops=[], cycles=0, rank_u=features.rank_u[node]
            )
            partitions.append(candidate)
        candidate.ops.append(node)
        candidate.cycles += int(op.get("cycles", 0))
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


def _partition_topology(partitions: list[Partition]) -> list[int]:
    preds = {p.id: set(p.preds) for p in partitions}
    succs = {p.id: set(p.succs) for p in partitions}
    return _topological_order([p.id for p in partitions], preds, succs)


def schedule_partitions(
    partitions: list[Partition],
    features: GraphFeatures,
    num_cores: int,
    scenario: str = "q2",
) -> tuple[dict[int, int], list[list[int]]]:
    """Critical-path-first list schedule for identical cores."""
    if num_cores < 1:
        raise ValueError("num_cores must be positive")
    if scenario not in {"q1", "q2", "q3"}:
        raise ValueError("scenario must be q1, q2, or q3")

    by_id = {p.id: p for p in partitions}
    topo = _partition_topology(partitions)
    topo_position = {pid: index for index, pid in enumerate(topo)}
    remaining = {p.id: len(p.preds) for p in partitions}
    ready = [p.id for p in partitions if remaining[p.id] == 0]
    core_time = [0] * num_cores
    finish: dict[int, int] = {}
    core_of: dict[int, int] = {}
    core_orders: list[list[int]] = [[] for _ in range(num_cores)]
    bandwidth = 60

    while ready:
        ready.sort(key=lambda pid: (-by_id[pid].rank_u, topo_position[pid]))
        pid = ready.pop(0)
        partition = by_id[pid]
        best: tuple[int, int, int] | None = None
        for core in range(num_cores):
            dependency_ready = 0
            for pred in partition.preds:
                delay = 0
                if scenario == "q1":
                    delay = 100 + math.ceil(_partition_edge_size(partitions, features, pred, pid) / bandwidth)
                elif core_of.get(pred) != core:
                    delay = 500 + math.ceil(_partition_edge_size(partitions, features, pred, pid) / bandwidth)
                dependency_ready = max(dependency_ready, finish[pred] + delay)
            start = max(core_time[core], dependency_ready)
            end = start + partition.cycles
            choice = (end, core_time[core], core)
            if best is None or choice < best:
                best = choice
        assert best is not None
        end, _, core = best
        core_of[pid] = core
        core_time[core] = end
        finish[pid] = end
        core_orders[core].append(pid)
        for successor in partition.succs:
            remaining[successor] -= 1
            if remaining[successor] == 0:
                ready.append(successor)
    return core_of, core_orders


def _partition_edge_size(partitions: list[Partition], features: GraphFeatures, source: int, target: int) -> int:
    source_ops = partitions[source].ops
    target_ops = partitions[target].ops
    return max(
        (_edge_size(features, u, v) for u in source_ops for v in target_ops if v in features.succs[u]),
        default=0,
    )


def build_plan(graph: dict[str, Any], num_cores: int = 4, scenario: str = "q2") -> dict[str, Any]:
    features = analyze_graph(graph)
    partitions = build_partitions(features)
    core_of, core_orders = schedule_partitions(partitions, features, num_cores, scenario)
    node_to_subgraph = {
        str(op_id): partition.id
        for partition in partitions
        for op_id in partition.ops
    }
    # The scheduler emits partition ids in each core's critical-path order.
    # Empty cores are intentionally retained in the output.
    return {
        "node_to_subgraph": node_to_subgraph,
        "core_schedules": core_orders,
    }


def load_graph(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        graph = json.load(stream)
    if not isinstance(graph, dict):
        raise ValueError("graph JSON must contain an object")
    return graph


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="极简统一计算图切分与多核调度框架")
    parser.add_argument("graph", type=Path)
    parser.add_argument("-n", "--num-cores", type=int, default=4)
    parser.add_argument("--scenario", choices=("q1", "q2", "q3"), default="q2")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args(argv)
    graph = load_graph(args.graph)
    plan = build_plan(graph, args.num_cores, args.scenario)
    output = args.output or args.graph.with_name(f"{args.graph.stem}_multicore_res.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        f"OK: scenario={args.scenario}, ops={len(plan['node_to_subgraph'])}, "
        f"subgraphs={len({*plan['node_to_subgraph'].values()})}, output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
