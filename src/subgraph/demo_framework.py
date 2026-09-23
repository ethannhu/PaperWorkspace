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
    )


def _edge_size(features: GraphFeatures, source: int, target: int) -> int:
    # Missing edge metadata means there is no known tensor transfer.  Falling
    # back to the largest tensor in the whole graph severely overestimates
    # unrelated dependencies after COPY contraction.
    return features.edge_sizes.get((source, target), 0)


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
        pipe = str(op.get("pipe", "UNKNOWN"))
        candidate.pipe_cycles[pipe] = candidate.pipe_cycles.get(pipe, 0) + int(
            op.get("cycles", 0)
        )
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
    # Scene-A uses a cheaper same-core wait than a cross-core wait.  These are
    # the fixed evaluator values from config.txt.
    same_core_wait = 100
    cross_core_wait = 1000
    core_pipe_load: list[dict[str, int]] = [dict() for _ in range(num_cores)]
    estimated_ddr_bytes = 0

    while ready:
        ready.sort(key=lambda pid: (-by_id[pid].rank_u, topo_position[pid]))
        pid = ready.pop(0)
        partition = by_id[pid]
        best: tuple[float, float, float, int, int] | None = None
        for core in range(num_cores):
            dependency_ready = 0
            candidate_edge_bytes = 0
            for pred in partition.preds:
                delay = 0
                if scenario == "q1":
                    edge_bytes = _partition_edge_size(partitions, features, pred, pid)
                    candidate_edge_bytes += edge_bytes
                    delay = (
                        same_core_wait if core_of.get(pred) == core else cross_core_wait
                    ) + math.ceil(edge_bytes / bandwidth)
                elif core_of.get(pred) != core:
                    edge_bytes = _partition_edge_size(partitions, features, pred, pid)
                    candidate_edge_bytes += edge_bytes
                    delay = 500 + math.ceil(edge_bytes / bandwidth)
                dependency_ready = max(dependency_ready, finish[pred] + delay)
            start = max(core_time[core], dependency_ready)
            end = start + partition.cycles
            # Pipe work is a lower-bound signal for the evaluator's overlapped
            # intra-core execution.  Keep the conservative total-cycle end
            # estimate, but prefer placements with lower per-pipe pressure.
            pipe_load = dict(core_pipe_load[core])
            for pipe, work in partition.pipe_cycles.items():
                pipe_load[pipe] = pipe_load.get(pipe, 0) + work
            max_pipe_load = max(pipe_load.values(), default=0)
            total_pipe_work = sum(pipe_load.values())
            avg_pipe_load = total_pipe_work / max(1, len(pipe_load))
            pipe_imbalance = max_pipe_load / max(1.0, avg_pipe_load)
            ddr_lb = (estimated_ddr_bytes + candidate_edge_bytes) / bandwidth
            # The lower bound cannot be hidden below the compute estimate.
            objective_end = max(float(end), ddr_lb)
            choice = (objective_end, pipe_imbalance, max_pipe_load, end, core)
            if best is None or choice < best:
                best = choice
        assert best is not None
        _, _, _, end, core = best
        candidate_edge_bytes = 0
        for pred in partition.preds:
            if scenario == "q1" or core_of.get(pred) != core:
                candidate_edge_bytes += _partition_edge_size(
                    partitions, features, pred, pid
                )
        estimated_ddr_bytes += candidate_edge_bytes
        core_of[pid] = core
        core_time[core] = end
        finish[pid] = end
        for pipe, work in partition.pipe_cycles.items():
            core_pipe_load[core][pipe] = core_pipe_load[core].get(pipe, 0) + work
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
