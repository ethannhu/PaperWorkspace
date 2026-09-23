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


@dataclass(frozen=True)
class CandidateDiagnostics:
    """Static diagnostics for one hypothetical partition placement."""

    partition_id: int | None
    candidate_core: int | None
    cross_core_edge_bytes: int
    critical_cross_core_edges: tuple[dict[str, Any], ...]
    per_core_compute_load: dict[int, int]
    per_core_partition_count: dict[int, int]
    objective_end: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "partition_id": self.partition_id,
            "candidate_core": self.candidate_core,
            "cross_core_edge_bytes": self.cross_core_edge_bytes,
            "critical_cross_core_edges": [dict(edge) for edge in self.critical_cross_core_edges],
            "per_core_compute_load": dict(self.per_core_compute_load),
            "per_core_partition_count": dict(self.per_core_partition_count),
            "objective_end": self.objective_end,
        }


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


def build_partitions(
    features: GraphFeatures,
    max_ops: int = 16,
    max_cycles: int = 20000,
    algorithm: str = "semantic",
) -> list[Partition]:
    """Build partitions with a selectable partitioning implementation."""
    if algorithm == "semantic":
        from .semantic_partition import semantic_partition

        return semantic_partition(features, max_ops=max_ops, max_cycles=max_cycles)
    if algorithm == "naive":
        from .naive_partition import naive_partition

        return naive_partition(features, max_ops=max_ops, max_cycles=max_cycles)
    raise ValueError("partition algorithm must be 'semantic' or 'naive'")


def _partition_topology(partitions: list[Partition]) -> list[int]:
    preds = {p.id: set(p.preds) for p in partitions}
    succs = {p.id: set(p.succs) for p in partitions}
    return _topological_order([p.id for p in partitions], preds, succs)


def _candidate_diagnostics(
    partitions: list[Partition],
    features: GraphFeatures,
    core_of: dict[int, int],
    num_cores: int,
    partition_id: int | None = None,
    candidate_core: int | None = None,
    objective_end: float | None = None,
) -> CandidateDiagnostics:
    """Measure communication and load for a complete or partial placement."""
    cross_core_edges: list[dict[str, Any]] = []
    cross_core_edge_bytes = 0
    for source in partitions:
        if source.id not in core_of:
            continue
        for target_id in sorted(source.succs):
            if target_id not in core_of or core_of[target_id] == core_of[source.id]:
                continue
            target = partitions[target_id]
            edge_bytes = _partition_transfer_bytes(features, source, target)
            criticality = max(source.rank_u, target.rank_u)
            cross_core_edge_bytes += edge_bytes
            cross_core_edges.append({
                "source_partition": source.id,
                "target_partition": target.id,
                "source_core": core_of[source.id],
                "target_core": core_of[target.id],
                "bytes": edge_bytes,
                "criticality": criticality,
            })
    cross_core_edges.sort(
        key=lambda edge: (-edge["bytes"], -edge["criticality"],
                          edge["source_partition"], edge["target_partition"])
    )
    compute_load = {core: 0 for core in range(num_cores)}
    partition_count = {core: 0 for core in range(num_cores)}
    by_id = {partition.id: partition for partition in partitions}
    for pid, core in core_of.items():
        compute_load[core] += by_id[pid].cycles
        partition_count[core] += 1
    return CandidateDiagnostics(
        partition_id=partition_id,
        candidate_core=candidate_core,
        cross_core_edge_bytes=cross_core_edge_bytes,
        # Keep the report compact while retaining the heaviest and most
        # critical communication boundaries for each candidate.
        critical_cross_core_edges=tuple(cross_core_edges[:10]),
        per_core_compute_load=compute_load,
        per_core_partition_count=partition_count,
        objective_end=objective_end,
    )


def schedule_partitions(
    partitions: list[Partition],
    features: GraphFeatures,
    num_cores: int,
    scenario: str = "q2",
    return_diagnostics: bool = False,
) -> tuple[dict[int, int], list[list[int]]] | tuple[
    dict[int, int], list[list[int]], dict[str, Any]
]:
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
    candidate_diagnostics: list[dict[str, Any]] = []

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
            if return_diagnostics:
                candidate_core_of = dict(core_of)
                candidate_core_of[pid] = core
                candidate_diagnostics.append(
                    _candidate_diagnostics(
                        partitions,
                        features,
                        candidate_core_of,
                        num_cores,
                        partition_id=pid,
                        candidate_core=core,
                        objective_end=objective_end,
                    ).as_dict()
                )
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
    if not return_diagnostics:
        return core_of, core_orders
    selected = _candidate_diagnostics(
        partitions,
        features,
        core_of,
        num_cores,
    ).as_dict()
    return core_of, core_orders, {
        "candidates": candidate_diagnostics,
        "selected": selected,
    }


def _partition_edge_size(partitions: list[Partition], features: GraphFeatures, source: int, target: int) -> int:
    source_ops = partitions[source].ops
    target_ops = partitions[target].ops
    return max(
        (_edge_size(features, u, v) for u in source_ops for v in target_ops if v in features.succs[u]),
        default=0,
    )


def _partition_transfer_bytes(
    features: GraphFeatures,
    source: Partition,
    target: Partition,
) -> int:
    """Sum each tensor crossing a partition pair once.

    A tensor can feed several operations in the destination partition.  The
    evaluator transfers that tensor once per remote task, so counting every
    operation pair would overstate the communication volume.
    """
    produced = set().union(
        *(features.output_tensors.get(op_id, set()) for op_id in source.ops)
    )
    consumed = set().union(
        *(features.input_tensors.get(op_id, set()) for op_id in target.ops)
    )
    return sum(
        int(features.tensor_by_id[tensor_id].get("size", 0))
        for tensor_id in produced & consumed
    )


def _core_orders_are_valid(
    partitions: list[Partition], core_orders: list[list[int]]
) -> tuple[list[int], dict[tuple[int, int], bool]] | None:
    """Return a topological order after adding per-core order constraints.

    The evaluator requires both the partition DAG and each core schedule to be
    acyclic.  A move that is locally harmless can still create a cycle through
    a cross-core dependency, so every local-search candidate goes through this
    small exact check.
    """
    by_id = {partition.id: partition for partition in partitions}
    all_ids = set(by_id)
    if set(pid for order in core_orders for pid in order) != all_ids:
        return None
    if any(len(order) != len(set(order)) for order in core_orders):
        return None

    succs = {pid: set(by_id[pid].succs) for pid in all_ids}
    preds = {pid: set(by_id[pid].preds) for pid in all_ids}
    original_edges: dict[tuple[int, int], bool] = {}
    for source in all_ids:
        for target in by_id[source].succs:
            original_edges[(source, target)] = True

    # Consecutive tasks on one core are ordered, but do not carry tensor data.
    for order in core_orders:
        for source, target in zip(order, order[1:]):
            if target not in succs[source]:
                succs[source].add(target)
                preds[target].add(source)
            original_edges.setdefault((source, target), False)

    ready = sorted(pid for pid in all_ids if not preds[pid])
    topo: list[int] = []
    while ready:
        pid = ready.pop(0)
        topo.append(pid)
        for target in sorted(succs[pid]):
            preds[target].remove(pid)
            if not preds[target]:
                ready.append(target)
                ready.sort()
    if len(topo) != len(all_ids):
        return None
    return topo, original_edges


def _proxy_schedule_objective(
    partitions: list[Partition],
    features: GraphFeatures,
    core_orders: list[list[int]],
    num_cores: int,
    scenario: str,
    transfer_bytes: dict[tuple[int, int], int] | None = None,
) -> tuple[float, int, int] | None:
    """Score one complete placement using the evaluator's fixed delays.

    This is intentionally a cheap filter, not a replacement for the official
    evaluator.  The tuple prioritizes proxy makespan, then communication, then
    compute imbalance.
    """
    checked = _core_orders_are_valid(partitions, core_orders)
    if checked is None:
        return None
    topo, edge_kinds = checked
    core_of = {
        pid: core
        for core, order in enumerate(core_orders)
        for pid in order
    }
    by_id = {partition.id: partition for partition in partitions}
    transfer_bytes = transfer_bytes or {
        (source.id, target.id): _partition_transfer_bytes(features, source, by_id[target_id])
        for source in partitions
        for target_id in source.succs
        for target in [by_id[target_id]]
    }
    artificial_preds: dict[int, list[int]] = {pid: [] for pid in by_id}
    for (source, target), is_original in edge_kinds.items():
        if not is_original:
            artificial_preds[target].append(source)
    finish: dict[int, int] = {}
    traffic = 0
    bandwidth = 60
    for pid in topo:
        start = 0
        for pred in by_id[pid].preds:
            edge_bytes = transfer_bytes.get((pred, pid), 0)
            same_core = core_of[pred] == core_of[pid]
            if scenario == "q1":
                delay = (100 if same_core else 1000) + math.ceil(edge_bytes / bandwidth)
                traffic += edge_bytes
            elif not same_core:
                delay = 500 + math.ceil(edge_bytes / bandwidth)
                traffic += edge_bytes
            else:
                delay = 0
            start = max(start, finish[pred] + delay)
        # Artificial same-core sequence edges have no tensor dependency and
        # are already represented by the predecessor finish time.
        for pred in artificial_preds[pid]:
            start = max(start, finish[pred])
        finish[pid] = start + by_id[pid].cycles

    loads = [sum(by_id[pid].cycles for pid in order) for order in core_orders]
    return max(finish.values(), default=0), traffic, max(loads, default=0) - min(loads, default=0)


def _local_search(
    partitions: list[Partition],
    features: GraphFeatures,
    core_of: dict[int, int],
    core_orders: list[list[int]],
    num_cores: int,
    scenario: str,
    max_iterations: int = 4,
) -> tuple[dict[int, int], list[list[int]], dict[str, Any]]:
    """Run a small deterministic best-improvement move/swap search."""
    if num_cores < 2 or len(partitions) < 2 or max_iterations <= 0:
        return core_of, core_orders, {"iterations": 0, "moves": 0}

    current = [list(order) for order in core_orders]
    by_id = {partition.id: partition for partition in partitions}
    transfer_bytes = {
        (source.id, target.id): _partition_transfer_bytes(features, source, target)
        for source in partitions
        for target_id in source.succs
        for target in [by_id[target_id]]
    }
    current_score = _proxy_schedule_objective(
        partitions, features, current, num_cores, scenario, transfer_bytes
    )
    if current_score is None:
        return core_of, core_orders, {"iterations": 0, "moves": 0, "invalid_initial": True}

    moves = 0
    iterations = 0
    by_id = {partition.id: partition for partition in partitions}
    for _ in range(max_iterations):
        iterations += 1
        loads = [sum(by_id[pid].cycles for pid in order) for order in current]
        busiest = max(range(num_cores), key=lambda core: (loads[core], -core))
        candidate_best: tuple[tuple[float, int, int], list[list[int]], str] | None = None

        # Move one task from the busiest core to another core.  Appending is
        # sufficient for the minimal search; the validity check rejects cycles.
        # Search the largest/most critical tasks first and cap the candidate
        # set so the cheap local improvement remains practical on large DAGs.
        candidate_pids = sorted(
            current[busiest],
            key=lambda pid: (-by_id[pid].cycles, -by_id[pid].rank_u, pid),
        )[:12]
        for pid in candidate_pids:
            for target in range(num_cores):
                if target == busiest:
                    continue
                candidate = [list(order) for order in current]
                candidate[busiest].remove(pid)
                candidate[target].append(pid)
                score = _proxy_schedule_objective(
                    partitions, features, candidate, num_cores, scenario, transfer_bytes
                )
                if score is not None and score < current_score:
                    item = (score, candidate, f"move:{pid}:{busiest}->{target}")
                    if candidate_best is None or item[0] < candidate_best[0]:
                        candidate_best = item

        # Also try pairwise swaps involving the busiest core.  This often
        # fixes a bad critical-path placement without changing core loads.
        for left_pid in candidate_pids:
            for target in range(num_cores):
                if target == busiest:
                    continue
                right_pids = sorted(
                    current[target],
                    key=lambda pid: (-by_id[pid].cycles, -by_id[pid].rank_u, pid),
                )[:8]
                for right_pid in right_pids:
                    candidate = [list(order) for order in current]
                    left_index = candidate[busiest].index(left_pid)
                    right_index = candidate[target].index(right_pid)
                    candidate[busiest][left_index] = right_pid
                    candidate[target][right_index] = left_pid
                    score = _proxy_schedule_objective(
                        partitions, features, candidate, num_cores, scenario, transfer_bytes
                    )
                    if score is not None and score < current_score:
                        item = (score, candidate, f"swap:{left_pid}:{right_pid}")
                        if candidate_best is None or item[0] < candidate_best[0]:
                            candidate_best = item

        if candidate_best is None:
            break
        current_score, current, _ = candidate_best
        moves += 1

    selected_core_of = {
        pid: core
        for core, order in enumerate(current)
        for pid in order
    }
    return selected_core_of, current, {
        "iterations": iterations,
        "moves": moves,
        "proxy_score": current_score,
    }


LOCAL_SEARCH_ALGORITHMS = {
    "none": None,
    "move_swap": _local_search,
}


def build_plan(
    graph: dict[str, Any],
    num_cores: int = 4,
    scenario: str = "q2",
    include_diagnostics: bool = False,
    partition_algorithm: str = "semantic",
    local_search_algorithm: str = "none",
    local_search_iterations: int = 4,
    local_search: bool | None = None,
) -> dict[str, Any]:
    # Keep the old boolean as a compatibility alias while making the named
    # strategy the primary, pluggable interface.
    if local_search is not None:
        local_search_algorithm = "move_swap" if local_search else "none"
    if local_search_algorithm not in LOCAL_SEARCH_ALGORITHMS:
        available = ", ".join(sorted(LOCAL_SEARCH_ALGORITHMS))
        raise ValueError(f"unknown local search algorithm: {local_search_algorithm}; available: {available}")
    features = analyze_graph(graph)
    partitions = build_partitions(features, algorithm=partition_algorithm)
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
    search_diagnostics = {
        "algorithm": local_search_algorithm,
        "iterations": 0,
        "moves": 0,
        "enabled": local_search_algorithm != "none",
    }
    search_algorithm = LOCAL_SEARCH_ALGORITHMS[local_search_algorithm]
    if search_algorithm is not None:
        core_of, core_orders, search_diagnostics = search_algorithm(
            partitions,
            features,
            core_of,
            core_orders,
            num_cores,
            scenario,
            max_iterations=local_search_iterations,
        )
        search_diagnostics.setdefault("algorithm", local_search_algorithm)
        search_diagnostics.setdefault("enabled", True)
    node_to_subgraph = {
        str(op_id): partition.id
        for partition in partitions
        for op_id in partition.ops
    }
    # The scheduler emits partition ids in each core's critical-path order.
    # Empty cores are intentionally retained in the output.
    plan: dict[str, Any] = {
        "node_to_subgraph": node_to_subgraph,
        "core_schedules": core_orders,
    }
    if include_diagnostics:
        plan["diagnostics"] = diagnostics
        plan["diagnostics"]["local_search"] = search_diagnostics
    return plan


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
    parser.add_argument(
        "--partition-algorithm",
        choices=("semantic", "naive"),
        default="semantic",
        help="分组算法；默认 semantic，可选历史朴素算法 naive",
    )
    parser.add_argument(
        "--local-search",
        choices=tuple(LOCAL_SEARCH_ALGORITHMS),
        default="none",
        help="局部搜索策略；默认关闭，可选 move_swap",
    )
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        help="额外输出候选方案诊断 JSON；不改变标准方案文件格式",
    )
    args = parser.parse_args(argv)
    graph = load_graph(args.graph)
    plan = build_plan(
        graph,
        args.num_cores,
        args.scenario,
        include_diagnostics=args.diagnostics_output is not None,
        partition_algorithm=args.partition_algorithm,
        local_search_algorithm=args.local_search,
    )
    diagnostics = plan.pop("diagnostics", None)
    output = args.output or args.graph.with_name(f"{args.graph.stem}_multicore_res.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if args.diagnostics_output is not None and diagnostics is not None:
        args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
        with args.diagnostics_output.open("w", encoding="utf-8") as stream:
            json.dump(diagnostics, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(
        f"OK: scenario={args.scenario}, ops={len(plan['node_to_subgraph'])}, "
        f"subgraphs={len({*plan['node_to_subgraph'].values()})}, output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
