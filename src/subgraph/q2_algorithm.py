"""Q2 独立分区与调度算法。

Q2 负责普通多核执行场景；本文件包含自己的分区、调度和入口逻辑，
不依赖 Q1 或 Q3 的算法模块。
"""

from __future__ import annotations

import math
from typing import Any

from .algorithm_common import (
    GraphFeatures,
    Partition,
    _edge_size,
    _topological_order,
    analyze_graph,
)
from .interfaces import AlgorithmResult


def _semantic_plan(
    features: GraphFeatures,
    num_cores: int,
    scenario: str,
) -> tuple[list[Partition], list[list[int]], dict[str, Any]]:
    # Q2 的默认路径：语义分区之后进行通信感知的列表调度。
    """Current complete strategy: semantic blocks followed by list scheduling."""
    from .semantic_partition import semantic_partition

    partitions = semantic_partition(features)
    core_orders = _schedule_partitions(partitions, features, num_cores, scenario)
    return partitions, core_orders, {
        "strategy": "semantic",
        "partition_count": len(partitions),
    }


def _coalesce_topological_partitions(
    partitions: list[Partition],
    max_ops: int,
    max_cycles: int,
) -> list[Partition]:
    """Merge adjacent partition-DAG blocks without crossing topological order.

    The semantic merger deliberately refuses many fan-in/fan-out joins.  That
    is useful for ordinary graphs, but it leaves wide replicated graphs with
    one block per operator.  Consecutive blocks in a topological order can be
    coalesced safely: every edge between the resulting intervals still points
    forward, so the quotient graph remains acyclic.  The size limits preserve
    enough intra-core parallelism and keep live memory pressure bounded.
    """
    if not partitions:
        return []
    topo = _partition_topology(partitions)
    by_id = {partition.id: partition for partition in partitions}
    groups: list[list[Partition]] = []
    current: list[Partition] = []
    current_ops = 0
    current_cycles = 0
    for pid in topo:
        partition = by_id[pid]
        too_large = current and (
            current_ops + len(partition.ops) > max_ops
            or current_cycles + partition.cycles > max_cycles
        )
        if too_large:
            groups.append(current)
            current = []
            current_ops = 0
            current_cycles = 0
        current.append(partition)
        current_ops += len(partition.ops)
        current_cycles += partition.cycles
    if current:
        groups.append(current)

    owner = {
        partition.id: group_id
        for group_id, group in enumerate(groups)
        for partition in group
    }
    merged: list[Partition] = []
    for group_id, group in enumerate(groups):
        ops = [op_id for partition in group for op_id in partition.ops]
        pipe_cycles: dict[str, int] = {}
        for partition in group:
            for pipe, cycles in partition.pipe_cycles.items():
                pipe_cycles[pipe] = pipe_cycles.get(pipe, 0) + cycles
        merged.append(
            Partition(
                id=group_id,
                ops=ops,
                cycles=sum(partition.cycles for partition in group),
                rank_u=max(partition.rank_u for partition in group),
                pipe_cycles=pipe_cycles,
            )
        )
    for partition in partitions:
        source_group = owner[partition.id]
        for successor in partition.succs:
            target_group = owner[successor]
            if source_group == target_group:
                continue
            merged[source_group].succs.add(target_group)
            merged[target_group].preds.add(source_group)
    return merged


def _schedule_pipelined_topology(
    partitions: list[Partition],
    num_cores: int,
) -> list[list[int]]:
    """Spread a narrow dependency spine so successive blocks can overlap."""
    if num_cores < 1:
        raise ValueError("num_cores must be positive")
    orders = [[] for _ in range(num_cores)]
    for index, pid in enumerate(_partition_topology(partitions)):
        orders[index % num_cores].append(pid)
    return orders


def _wide_plan(
    features: GraphFeatures,
    num_cores: int,
    scenario: str,
    pattern: Any,
) -> tuple[list[Partition], list[list[int]], dict[str, Any]]:
    """Communication-aware strategy for wide replicated computation graphs.

    WIDE graphs expose abundant parallelism, but fan-in/fan-out prevents the
    strict semantic pass from fusing enough operators.  Coalesce its blocks in
    topological intervals, keeping compute/activation or gate stages local
    while preventing a single partition from becoming too large.
    """
    from .graph_patterns import GraphPattern
    from .semantic_partition import semantic_partition

    base = semantic_partition(features, max_ops=16, max_cycles=20000)
    operator_count = len(features.topo_order)
    if pattern == GraphPattern.WIDE_MATMUL_ADD:
        # A long, moderately wide ADD spine needs smaller blocks so the
        # scheduler can pipeline successive stages across cores.  Very wide
        # shallow batches benefit from larger communication-saving blocks.
        layer_width = max(
            (
                sum(features.depth[node] == level for node in features.topo_order)
                for level in set(features.depth.values())
            ),
            default=0,
        )
        if operator_count < 5000:
            max_ops, max_cycles = 16, 20000
            motif = "semantic_wide_fallback"
        elif max(features.depth.values(), default=0) >= 16 and layer_width < 1000:
            max_ops, max_cycles = 8, 6000
            motif = "matmul_add_pipelined_spine"
        else:
            max_ops, max_cycles = 32, 18000
            motif = "matmul_add_fan_in"
    elif pattern == GraphPattern.GATED_SIGMOID_MLP:
        max_ops, max_cycles = 24, 18000
        motif = "gated_chain"
    else:
        max_ops, max_cycles = 24, 16000
        motif = "compute_activation_branch"
    if pattern == GraphPattern.SHALLOW_WIDE_COMPUTE_ACTIVATION and operator_count < 5000:
        max_ops, max_cycles = 16, 20000
        motif = "semantic_wide_fallback"
    if motif == "semantic_wide_fallback":
        partitions = base
    else:
        partitions = _coalesce_topological_partitions(base, max_ops, max_cycles)
    if motif == "matmul_add_pipelined_spine":
        core_orders = _schedule_pipelined_topology(partitions, num_cores)
        scheduler = "round_robin_spine_pipeline"
    elif motif == "semantic_wide_fallback":
        core_orders = _schedule_partitions(partitions, features, num_cores, scenario)
        scheduler = "communication_aware_list"
    else:
        core_orders = _schedule_partitions(partitions, features, num_cores, scenario)
        scheduler = "communication_aware_list"
    return partitions, core_orders, {
        "strategy": "wide_communication_aware_coalescing",
        "scheduler": scheduler,
        "motif": motif,
        "base_partition_count": len(base),
        "partition_count": len(partitions),
        "max_ops": max_ops,
        "max_cycles": max_cycles,
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
    # 用依赖就绪时间、计算结束时间和管线负载共同选择目标核。
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
    cache_reuse = (
        _partition_cache_reuse(partitions, features)
        if scenario == "q3" and _q3_reuse_priority(features)
        else {}
    )

    while ready:
        ready.sort(
            key=lambda pid: (
                -by_id[pid].rank_u,
                -cache_reuse.get(pid, 0),
                topo_position[pid],
            )
        )
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
    cache_reuse = (
        _partition_cache_reuse(partitions, features)
        if scenario == "q3" and _q3_reuse_priority(features)
        else {}
    )

    while ready:
        ready.sort(
            key=lambda pid: (
                pid not in critical_set,
                critical_index.get(pid, len(critical_path)),
                -cache_reuse.get(pid, 0),
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


def _partition_cache_reuse(
    partitions: list[Partition], features: GraphFeatures
) -> dict[int, int]:
    """Estimate Q3 reuse value from tensors consumed by multiple partitions."""
    users: dict[int, set[int]] = {}
    inputs: dict[int, set[int]] = {}
    for partition in partitions:
        tensor_ids = {
            tensor_id
            for op_id in partition.ops
            for tensor_id in features.input_tensors.get(op_id, ())
        }
        inputs[partition.id] = tensor_ids
        for tensor_id in tensor_ids:
            users.setdefault(tensor_id, set()).add(partition.id)
    return {
        partition_id: sum(
            int(features.tensor_by_id[tensor_id].get("size", 0))
            * (len(users[tensor_id]) - 1)
            for tensor_id in tensor_ids
            if len(users[tensor_id]) > 1
        )
        for partition_id, tensor_ids in inputs.items()
    }


def _q3_reuse_priority(features: GraphFeatures) -> bool:
    """Enable reuse ordering only for branch-heavy replicated motifs."""
    from .graph_patterns import GraphPattern, classify_features

    return classify_features(features).pattern in {
        GraphPattern.CNN_RESIDUAL,
        GraphPattern.GATED_SIGMOID_MLP,
    }


def build_algorithm_plan(
    features: GraphFeatures,
    num_cores: int = 4,
    scenario: str = "q2",
) -> AlgorithmResult:
    # Q2 在本文件内完成图族识别、分区和调度，不调用其他题目的入口。
    """Build a plan with the complete strategy chosen by graph pattern."""
    from .graph_patterns import GraphPatternFamily, classify_features

    graph_pattern = classify_features(features)
    if graph_pattern.family == GraphPatternFamily.WIDE:
        partitions, core_orders, strategy_diagnostics = _wide_plan(
            features,
            num_cores,
            scenario,
            graph_pattern.pattern,
        )
    elif graph_pattern.family == GraphPatternFamily.COMPLEX:
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


def build_plan(
    graph: dict[str, Any],
    num_cores: int = 4,
    features: GraphFeatures | None = None,
) -> AlgorithmResult:
    # 对外统一入口：输入原始图，输出子图编号和每个核的执行顺序。
    """Build the Q2 plan using the current strategy implementation."""
    if features is None:
        features = analyze_graph(graph)
    return build_algorithm_plan(features, num_cores, scenario="q2")
