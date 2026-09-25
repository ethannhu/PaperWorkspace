"""Planning pipeline for graph partitioning and multicore scheduling.

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
from typing import Any

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


def _semantic_plan(
    features: GraphFeatures,
    num_cores: int,
    scenario: str,
) -> tuple[list[Partition], list[list[int]], dict[str, Any]]:
    """Current complete strategy: semantic blocks followed by list scheduling."""
    from .semantic_partition import semantic_partition

    partitions = semantic_partition(features)
    core_orders = _schedule_partitions(partitions, features, num_cores, scenario)
    return partitions, core_orders, {
        "strategy": "semantic",
        "partition_count": len(partitions),
    }


def _complex_plan(
    features: GraphFeatures,
    num_cores: int,
    scenario: str,
    pattern: Any,
) -> tuple[list[Partition], list[list[int]], dict[str, Any]]:
    """Strategy for residual/attention-like graphs with expensive joins.

    Complex graphs usually lose more from splitting a long residual or
    normalize chain than they gain from exposing tiny extra tasks.  Use the
    semantic partitioner with singleton repair, then keep the partition DAG's
    heaviest critical spine sticky during scheduling.
    """
    from .semantic_partition import semantic_partition

    partitions = semantic_partition(
        features,
        max_ops=24,
        max_cycles=50000,
        enable_singleton_repair=True,
    )
    critical_path = _critical_partition_path(partitions, features)
    critical_cycles = sum(partitions[pid].cycles for pid in critical_path)
    from .graph_patterns import GraphPattern

    # A narrow/deep CNN can be a replicated family of independent residual
    # branches.  Protecting one arbitrary branch with sticky affinity leaves
    # one core nearly idle and overloads another.  Preserve branch parallelism
    # for this shape; wider CNNs and truly serial residual spines keep the
    # critical-path scheduler.
    layer_width = max(
        (
            sum(features.depth[node] == level for node in features.topo_order)
            for level in set(features.depth.values())
        ),
        default=0,
    )
    replicated_narrow_cnn = (
        pattern == GraphPattern.CNN_RESIDUAL
        and max(features.depth.values(), default=0) >= 80
        and layer_width <= 16
    )
    if (
        replicated_narrow_cnn
        or pattern == GraphPattern.CNN_RESIDUAL and critical_cycles > 50000
    ):
        if replicated_narrow_cnn:
            core_orders = _schedule_replicated_components(
                partitions, num_cores
            )
            scheduler = "balanced_replicated_component_list"
        else:
            core_orders = _schedule_partitions(partitions, features, num_cores, scenario)
            scheduler = "plain_list_after_complex_fusion"
    else:
        core_orders = _schedule_complex_partitions(
            partitions,
            features,
            num_cores,
            scenario,
            critical_path,
        )
        scheduler = "critical_path_sticky"
    return partitions, core_orders, {
        "strategy": "complex_semantic_critical_path",
        "scheduler": scheduler,
        "partition_count": len(partitions),
        "critical_path_length": len(critical_path),
        "critical_path_cycles": critical_cycles,
        "max_layer_width": layer_width,
        "replicated_narrow_cnn": replicated_narrow_cnn,
    }


def _mixed_plan(
    features: GraphFeatures,
    num_cores: int,
    scenario: str,
) -> tuple[list[Partition], list[list[int]], dict[str, Any]]:
    """Plan the medium-sized mixed MLP/Reduce family.

    MIXED graphs have enough joins to make a purely local greedy fusion noisy,
    but not enough serial depth to justify the aggressive sticky policy used
    by attention and residual graphs.  Keep blocks smaller when the graph is
    wide/deep, repair only profitable singleton blocks, then protect one heavy
    partition path with a reduced affinity penalty.
    """
    from .semantic_partition import semantic_partition

    # The documented MIXED median is 34 levels / 152 nodes per level.  These
    # limits preserve branch parallelism while still fusing short compute,
    # activation, and reduction stages.
    wide_or_deep = (
        max(features.depth.values(), default=0) >= 60
        or max(
            (sum(features.depth[node] == level for node in features.topo_order)
             for level in set(features.depth.values())),
            default=0,
        ) >= 180
    )
    max_ops = 10 if wide_or_deep else 14
    max_cycles = 12000 if wide_or_deep else 18000
    partitions = semantic_partition(
        features,
        max_ops=max_ops,
        max_cycles=max_cycles,
        enable_singleton_repair=True,
    )
    critical_path = _critical_partition_path(partitions, features)
    critical_cycles = sum(partitions[pid].cycles for pid in critical_path)
    core_orders = _schedule_complex_partitions(
        partitions,
        features,
        num_cores,
        scenario,
        critical_path,
        split_penalty_scale=0.35,
    )
    return partitions, core_orders, {
        "strategy": "mixed_semantic_fusion_critical_path",
        "scheduler": "critical_path_soft_sticky",
        "partition_count": len(partitions),
        "critical_path_length": len(critical_path),
        "critical_path_cycles": critical_cycles,
        "max_ops": max_ops,
        "max_cycles": max_cycles,
        "singleton_repair": True,
    }


def _partition_topology(partitions: list[Partition]) -> list[int]:
    preds = {p.id: set(p.preds) for p in partitions}
    succs = {p.id: set(p.succs) for p in partitions}
    return _topological_order([p.id for p in partitions], preds, succs)


def _schedule_replicated_components(
    partitions: list[Partition],
    num_cores: int,
) -> list[list[int]]:
    """Pack disconnected branch components without splitting a branch.

    Narrow/deep residual inputs often contain replicated independent chains.
    Their partition DAG has no edges between replicas, so treating every
    partition as an independent ready task can interleave branches and create
    avoidable memory-pipeline dependencies.  Keep each component contiguous
    and use largest-processing-time-first packing across cores.
    """
    if num_cores < 1:
        raise ValueError("num_cores must be positive")
    if not partitions:
        return [[] for _ in range(num_cores)]

    by_id = {partition.id: partition for partition in partitions}
    neighbours = {
        partition.id: partition.preds | partition.succs
        for partition in partitions
    }
    components: list[list[int]] = []
    unseen = set(by_id)
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack = [start]
        component: list[int] = []
        while stack:
            pid = stack.pop()
            component.append(pid)
            for neighbour in neighbours[pid] & unseen:
                unseen.remove(neighbour)
                stack.append(neighbour)
        components.append(component)

    topo = _partition_topology(partitions)
    topo_index = {pid: index for index, pid in enumerate(topo)}
    ordered_components: list[tuple[int, list[int], int]] = []
    for component in components:
        ordered = sorted(component, key=topo_index.__getitem__)
        work = sum(by_id[pid].cycles for pid in ordered)
        ordered_components.append((work, ordered, min(ordered)))
    ordered_components.sort(key=lambda item: (-item[0], item[2]))

    assigned_components: list[list[list[int]]] = [
        [] for _ in range(num_cores)
    ]
    core_load = [0] * num_cores
    for work, component, _ in ordered_components:
        core = min(range(num_cores), key=lambda index: (core_load[index], index))
        assigned_components[core].append(component)
        core_load[core] += work

    # Keep replicas interleaved by stage.  Concatenating whole components
    # serializes one branch before touching the next and defeats Q2/Q3
    # same-core tensor reuse, especially when all components share a core.
    core_orders: list[list[int]] = []
    for components in assigned_components:
        order: list[int] = []
        for index in range(max((len(component) for component in components), default=0)):
            for component in components:
                if index < len(component):
                    order.append(component[index])
        core_orders.append(order)
    return core_orders


def _critical_partition_path(
    partitions: list[Partition],
    features: GraphFeatures,
) -> list[int]:
    """Return one heavy downstream path through the partition DAG."""
    if not partitions:
        return []
    by_id = {partition.id: partition for partition in partitions}
    topo = _partition_topology(partitions)
    score: dict[int, int] = {}
    next_on_path: dict[int, int] = {}
    for pid in reversed(topo):
        partition = by_id[pid]
        best_successor = None
        best_score = 0
        for successor in partition.succs:
            edge_cycles = math.ceil(
                _partition_edge_size(partitions, features, pid, successor) / 60
            )
            candidate = edge_cycles + score[successor]
            if candidate > best_score:
                best_score = candidate
                best_successor = successor
        score[pid] = partition.cycles + best_score
        if best_successor is not None:
            next_on_path[pid] = best_successor
    start = max(topo, key=lambda pid: (score[pid], by_id[pid].rank_u, -pid))
    path = [start]
    while path[-1] in next_on_path:
        path.append(next_on_path[path[-1]])
    return path


def _schedule_partitions(
    partitions: list[Partition],
    features: GraphFeatures,
    num_cores: int,
    scenario: str = "q2",
) -> list[list[int]]:
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
    return core_orders


def _schedule_complex_partitions(
    partitions: list[Partition],
    features: GraphFeatures,
    num_cores: int,
    scenario: str,
    critical_path: list[int],
    split_penalty_scale: float = 1.0,
) -> list[list[int]]:
    """Schedule complex graphs while keeping the critical spine coherent."""
    if num_cores < 1:
        raise ValueError("num_cores must be positive")
    if scenario not in {"q1", "q2", "q3"}:
        raise ValueError("scenario must be q1, q2, or q3")

    by_id = {p.id: p for p in partitions}
    topo = _partition_topology(partitions)
    topo_position = {pid: index for index, pid in enumerate(topo)}
    critical_index = {pid: index for index, pid in enumerate(critical_path)}
    critical_set = set(critical_path)
    critical_predecessor = {
        critical_path[index]: critical_path[index - 1]
        for index in range(1, len(critical_path))
    }
    remaining = {p.id: len(p.preds) for p in partitions}
    ready = [p.id for p in partitions if remaining[p.id] == 0]
    core_time = [0] * num_cores
    finish: dict[int, int] = {}
    core_of: dict[int, int] = {}
    core_orders: list[list[int]] = [[] for _ in range(num_cores)]
    core_pipe_load: list[dict[str, int]] = [dict() for _ in range(num_cores)]
    estimated_ddr_bytes = 0
    bandwidth = 60

    while ready:
        ready.sort(
            key=lambda pid: (
                pid not in critical_set,
                critical_index.get(pid, len(critical_path)),
                -by_id[pid].rank_u,
                topo_position[pid],
            )
        )
        pid = ready.pop(0)
        partition = by_id[pid]
        sticky_core = _preferred_complex_core(
            partitions,
            features,
            pid,
            core_of,
            critical_predecessor,
        )
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
                        100 if core_of.get(pred) == core else 1000
                    ) + math.ceil(edge_bytes / bandwidth)
                elif core_of.get(pred) != core:
                    edge_bytes = _partition_edge_size(partitions, features, pred, pid)
                    candidate_edge_bytes += edge_bytes
                    delay = 500 + math.ceil(edge_bytes / bandwidth)
                dependency_ready = max(dependency_ready, finish[pred] + delay)
            start = max(core_time[core], dependency_ready)
            end = start + partition.cycles
            pipe_load = dict(core_pipe_load[core])
            for pipe, work in partition.pipe_cycles.items():
                pipe_load[pipe] = pipe_load.get(pipe, 0) + work
            max_pipe_load = max(pipe_load.values(), default=0)
            total_pipe_work = sum(pipe_load.values())
            avg_pipe_load = total_pipe_work / max(1, len(pipe_load))
            pipe_imbalance = max_pipe_load / max(1.0, avg_pipe_load)
            ddr_lb = (estimated_ddr_bytes + candidate_edge_bytes) / bandwidth
            objective_end = max(float(end), ddr_lb)
            if sticky_core is not None and core != sticky_core:
                objective_end += _complex_split_penalty(
                    partitions,
                    features,
                    pid,
                    core_of,
                    scenario,
                    scale=split_penalty_scale,
                )
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
    return core_orders


def _preferred_complex_core(
    partitions: list[Partition],
    features: GraphFeatures,
    pid: int,
    core_of: dict[int, int],
    critical_predecessor: dict[int, int],
) -> int | None:
    """Pick the predecessor core worth preserving for a complex-graph join."""
    critical_pred = critical_predecessor.get(pid)
    if critical_pred is not None and critical_pred in core_of:
        return core_of[critical_pred]
    placed_preds = [pred for pred in partitions[pid].preds if pred in core_of]
    if not placed_preds:
        return None
    heavy_pred = max(
        placed_preds,
        key=lambda pred: (
            _partition_edge_size(partitions, features, pred, pid),
            partitions[pred].rank_u,
            -pred,
        ),
    )
    edge_bytes = _partition_edge_size(partitions, features, heavy_pred, pid)
    if len(partitions[pid].preds) > 1 or edge_bytes > 0:
        return core_of[heavy_pred]
    return None


def _complex_split_penalty(
    partitions: list[Partition],
    features: GraphFeatures,
    pid: int,
    core_of: dict[int, int],
    scenario: str,
    scale: float = 1.0,
) -> float:
    """Extra cost for splitting protected complex-chain dependencies."""
    edge_bytes = max(
        (
            _partition_edge_size(partitions, features, pred, pid)
            for pred in partitions[pid].preds
            if pred in core_of
        ),
        default=0,
    )
    transfer = math.ceil(edge_bytes / 60)
    fixed = 1000 if scenario == "q1" else 500
    return float(scale * (fixed + transfer))


def _partition_edge_size(partitions: list[Partition], features: GraphFeatures, source: int, target: int) -> int:
    source_ops = partitions[source].ops
    target_ops = partitions[target].ops
    return max(
        (_edge_size(features, u, v) for u in source_ops for v in target_ops if v in features.succs[u]),
        default=0,
    )


def build_algorithm_plan(
    features: GraphFeatures,
    num_cores: int = 4,
    scenario: str = "q2",
) -> AlgorithmResult:
    """Build a plan with the complete strategy chosen by graph pattern."""
    from .graph_patterns import GraphPatternFamily, classify_features

    graph_pattern = classify_features(features)
    if graph_pattern.family == GraphPatternFamily.COMPLEX:
        partitions, core_orders, strategy_diagnostics = _complex_plan(
            features,
            num_cores,
            scenario,
            graph_pattern.pattern,
        )
    elif graph_pattern.family == GraphPatternFamily.MIXED:
        partitions, core_orders, strategy_diagnostics = _mixed_plan(
            features,
            num_cores,
            scenario,
        )
    else:
        partitions, core_orders, strategy_diagnostics = _semantic_plan(
            features,
            num_cores,
            scenario,
        )
    node_to_subgraph = {
        str(op_id): partition.id
        for partition in partitions
        for op_id in partition.ops
    }
    # Empty cores are intentionally retained in the output.
    return AlgorithmResult(
        plan={
            "node_to_subgraph": node_to_subgraph,
            "core_schedules": core_orders,
        },
        diagnostics={
            "graph_pattern": graph_pattern.as_dict(),
            "algorithm": strategy_diagnostics,
        },
    )
