# ============================================================
# 人工智能工具信息 | AI Tool Information
#   工具名称 (Tool Name)        : GLM-5.2
#   版本/型号 (Version/Model)   : GLM-5.2
#   开发机构/公司 (Developer)    : 智谱AI (Zhipu AI / zai-org)
#   版本颁布日期 (Release Date) : 2026-06-13
#   声明：本程序及代码是在人工智能工具辅助下完成的
# ============================================================
"""Q3 独立分区与调度算法。

Q3 在本文件内维护完整的分区、调度和缓存感知策略，使用 Q3 的缓存模型。
本文件不引用 Q1/Q2，便于单独实验和后续维护。

Q3 的成本模型（与 Q2 一样属 scene-B）：
    * 跨核固定等待 = 500 cycles；
    * 跨核字节传输带宽 = 60 bytes/cycle；
    * 额外启用 L2 只读 FIFO 缓存：``q3_cache_mode = "read_only_fifo"``。

Q3 相对 Q2 的核心增量是 **缓存复用优先级**：对分支重图族（CNN_RESIDUAL
和 GATED_SIGMOID_MLP），调度器 ready 排序里加入“被多分区共享的张量字节数”
作为第二优先级，并据此把高复用分区排到前面以提升 L2 命中率（见
``_partition_cache_reuse`` 与 ``_q3_reuse_priority``）。

支持五组消融：
    * no_l2                  —— 关闭 L2 复用（方案写入 cache_mode="disabled"）；
    * no_cache_priority      —— 关闭缓存优先级（但保留 L2）；
    * global_cache_priority  —— 强制所有图族启用缓存优先级；
    * no_communication       —— 关闭跨核通信感知；
    * full                   —— 完整策略。
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
    rebalance_core_orders,
)
from .interfaces import AlgorithmResult

# None 表示完整策略；其余字符串为消融开关。
ABLATIONS = {None, "no_l2", "no_cache_priority", "global_cache_priority", "no_communication"}


def _semantic_plan(
    features: GraphFeatures,
    num_cores: int,
    scenario: str,
) -> tuple[list[Partition], list[list[int]], dict[str, Any]]:
    """默认完整策略：语义分块 + Q3 缓存感知列表调度。"""
    # Q3 保留语义分区，同时把缓存复用信息交给本文件的调度逻辑。
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
    """按拓扑序合并相邻分区块，不跨越拓扑方向。

    语义合并器刻意拒绝许多 fan-in/fan-out 汇合。对普通图这是合理的，但会让
    宽复制图退化为“每个算子一块”。在拓扑序上把连续块直接拼接是安全的：合并
    区间之间的所有边仍然正向，商图仍然无环。``max_ops`` / ``max_cycles`` 上
    限既保留核内并行度，也限制活跃内存压力。
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
    """把窄依赖主链按轮转散开，让相邻块跨核重叠执行。"""
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
    """针对“宽并行复制图”的通信感知策略。

    WIDE 图天然有充足并行度，但 fan-in/fan-out 阻挡了严格语义合并凑出足够大
    的算子组。在拓扑区间上把块再合并，使 compute/activation 或门控 stage 保
    持本地，同时防止单个分区过大失去并行度。
    """
    from .graph_patterns import GraphPattern
    from .semantic_partition import semantic_partition

    operator_count = len(features.topo_order)
    if pattern == GraphPattern.WIDE_MATMUL_ADD:
        # 让重复的 MatMul-Add tile 保留在本地（与 Q1 一致），消除之前在 Q3
        # scene-B 中占主导的多余边界；下方仍走 Q3 的缓存感知调度。
        layer_width = max(
            (
                sum(features.depth[node] == level for node in features.topo_order)
                for level in set(features.depth.values())
            ),
            default=0,
        )
        depth = max(features.depth.values(), default=0)
        max_ops, max_cycles = (24, 32000) if layer_width < 1000 else (20, 24000)
        if depth >= 16 and layer_width < 1000:
            max_ops, max_cycles = 14, 14000
        motif = "matmul_add_boundary_fusion"
        base = semantic_partition(
            features,
            max_ops=max_ops,
            max_cycles=max_cycles,
            enable_singleton_repair=True,
        )
        partitions = _coalesce_topological_partitions(
            base, min(max_ops, 32), min(max_cycles, 50000)
        )
        core_orders = _schedule_partitions(partitions, features, num_cores, scenario)
        scheduler = "q3_affinity_list"
    elif pattern == GraphPattern.GATED_SIGMOID_MLP:
        base = semantic_partition(features, max_ops=16, max_cycles=20000)
        max_ops, max_cycles = 24, 18000
        motif = "gated_chain"
    else:
        base = semantic_partition(features, max_ops=16, max_cycles=20000)
        max_ops, max_cycles = 24, 16000
        motif = "compute_activation_branch"
    if pattern == GraphPattern.SHALLOW_WIDE_COMPUTE_ACTIVATION and operator_count < 5000:
        # 浅宽且规模不大 → 退回纯语义分块。
        base = semantic_partition(features, max_ops=16, max_cycles=20000)
        max_ops, max_cycles = 16, 20000
        motif = "semantic_wide_fallback"
    if pattern != GraphPattern.WIDE_MATMUL_ADD:
        if motif == "semantic_wide_fallback":
            partitions = base
        else:
            partitions = _coalesce_topological_partitions(base, max_ops, max_cycles)
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
    """针对残差/Attention 这类“复杂图”的策略。

    复杂图通常从拆分长残差/归一化链路中损失更多，远比暴露几个小 task 得到
    的收益多。语义分区器开启 singleton 修复，并把分区 DAG 上最重的关键主链
    用粘性调度保持连贯。
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

    # 窄而深的 CNN 可能是若干独立的残差分支副本。给一条任意分支上关键路径
    # 粘性会让一个核接近空闲、其他核过载。对这种形态要保留分支并行度；更宽
    # 的 CNN 与真正串行的残差主链则继续走关键路径调度。
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
            # 副本链小且传输受限：case_044 有 11 条 7672-cycle 链，超过 2 核
            # 会让 COPY_IN 流量超过节省的计算。更长副本（case_046/078）则保
            # 留完整核数。
            effective_cores = (
                min(num_cores, 2) if critical_cycles <= 10000 else num_cores
            )
            core_orders = _schedule_replicated_components(
                partitions, effective_cores
            )
            core_orders.extend([[] for _ in range(num_cores - effective_cores)])
            scheduler = "balanced_replicated_component_list"
        else:
            core_orders = _schedule_partitions(partitions, features, num_cores, scenario)
            scheduler = "plain_list_after_complex_fusion"
    else:
        # Q3 对 COMPLEX 家族默认开启关键路径粘性 + singleton 修复。
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
        "effective_core_count": (
            min(num_cores, 2)
            if replicated_narrow_cnn and critical_cycles <= 10000
            else num_cores
        ),
    }


def _mixed_plan(
    features: GraphFeatures,
    num_cores: int,
    scenario: str,
) -> tuple[list[Partition], list[list[int]], dict[str, Any]]:
    """针对“中等规模混合 MLP/Reduce”图族的策略。

    MIXED 图的汇合数足够多以致纯贪心合并会很嘈杂，但又没有足够串行深度去
    支持 attention/残差图的强力粘性策略。当图宽/图深更大时使用更小的块，
    仅修复有收益的 singleton，然后用更低的粘性惩罚保护一条重分区路径。
    """
    from .semantic_partition import semantic_partition

    # 文档标注的 MIXED 中位数是 34 层 / 每层 152 节点。下面的上限在保留分支
    # 并行的同时仍能融合短 compute / activation / reduce stage。
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
    # MIXED 与 Q1 同样使用 0.35 的柔和粘性，没有 critical_affinity 开关。
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
    """对分区 DAG 跑一次拓扑排序。"""
    preds = {p.id: set(p.preds) for p in partitions}
    succs = {p.id: set(p.succs) for p in partitions}
    return _topological_order([p.id for p in partitions], preds, succs)


def _schedule_replicated_components(
    partitions: list[Partition],
    num_cores: int,
) -> list[list[int]]:
    """把不连通的分支组件按“整条分支”打包，不拆分支。

    窄/深残差输入常常包含若干独立链路副本。它们的分区 DAG 中副本之间没有
    边，把每个分区当独立就绪 task 处理会让分支交错，制造可避免的内存管线依
    赖。每个组件保持连续，按 LPT（最大处理时间优先）打包到各核。
    """
    if num_cores < 1:
        raise ValueError("num_cores must be positive")
    if not partitions:
        return [[] for _ in range(num_cores)]

    # 1) 找连通分量：neighbours = preds ∪ succs。
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

    # 2) 组件内部按拓扑序；组件之间按 work 降序（LPT）打包。
    topo = _partition_topology(partitions)
    topo_index = {pid: index for index, pid in enumerate(topo)}
    ordered_components: list[tuple[int, list[int], int]] = []
    for component in components:
        ordered = sorted(component, key=topo_index.__getitem__)
        work = sum(by_id[pid].cycles for pid in ordered)
        ordered_components.append((work, ordered, min(ordered)))
    ordered_components.sort(key=lambda item: (-item[0], item[2]))

    # 3) LPT：把组件扔给当前负载最轻的核。
    assigned_components: list[list[list[int]]] = [
        [] for _ in range(num_cores)
    ]
    core_load = [0] * num_cores
    for work, component, _ in ordered_components:
        core = min(range(num_cores), key=lambda index: (core_load[index], index))
        assigned_components[core].append(component)
        core_load[core] += work

    # 4) 同核多副本按 stage 交错拼接，而不是连整条 branch。
    # 直接连整条会让一条 branch 整段跑完才轮到下一条，破坏 Q2/Q3 同核张量复
    # 用，特别是当多个副本落在同核时。
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
    """找出分区 DAG 上的一条重下游路径。"""
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
            # 把跨核边传输时间（按 60 bytes/cycle 折算）也算进路径分数。
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
    cache_priority: bool | None = None,
    communication_aware: bool = True,
) -> list[list[int]]:
    """Q3 缓存感知 + 关键路径优先的列表调度器。

    在依赖和负载约束下优先安排缓存复用价值高的分区。``cache_priority``：
        * None —— 仅对分支重图族启用（默认）；
        * True —— 强制所有图族启用（``global_cache_priority`` 消融）；
        * False —— 完全禁用缓存优先级（``no_cache_priority`` 消融，仍保留 L2）。
    其余与 Q2 调度器一致：``(objective_end, pipe_imbalance, max_pipe_load, end, core)``
    的目标元组决定核选择。
    """
    # 在依赖和负载约束下优先安排缓存复用价值高的分区。
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
    # scene-A 的同核等待便宜、跨核等待贵。这两个数固定取自 config.txt。
    same_core_wait = 100
    cross_core_wait = 1000
    core_pipe_load: list[dict[str, int]] = [dict() for _ in range(num_cores)]
    estimated_ddr_bytes = 0
    cache_reuse = (
        _partition_cache_reuse(partitions, features)
        if scenario == "q3" and _q3_reuse_priority(features, cache_priority)
        else {}
    )

    while ready:
        # 排序：rank_u 大优先；cache_reuse 大优先；拓扑位置小优先。
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
                elif communication_aware and core_of.get(pred) != core:
                    # scene-B：仅跨核才收 500 + bytes/60。
                    edge_bytes = _partition_edge_size(partitions, features, pred, pid)
                    candidate_edge_bytes += edge_bytes
                    delay = 500 + math.ceil(edge_bytes / bandwidth)
                dependency_ready = max(dependency_ready, finish[pred] + delay)
            start = max(core_time[core], dependency_ready)
            end = start + partition.cycles
            # Pipe 工作量是评测器核内重叠执行的下界信号。保留保守“总 end”估
            # 计，同时优先选管线压力更小的核。
            pipe_load = dict(core_pipe_load[core])
            for pipe, work in partition.pipe_cycles.items():
                pipe_load[pipe] = pipe_load.get(pipe, 0) + work
            max_pipe_load = max(pipe_load.values(), default=0)
            total_pipe_work = sum(pipe_load.values())
            avg_pipe_load = total_pipe_work / max(1, len(pipe_load))
            pipe_imbalance = max_pipe_load / max(1.0, avg_pipe_load)
            ddr_lb = (estimated_ddr_bytes + candidate_edge_bytes) / bandwidth if communication_aware else 0
            # DDR 下界无法藏在计算时间下面，所以目标 end 取 max(end, ddr_lb)。
            objective_end = max(float(end), ddr_lb)
            choice = (objective_end, pipe_imbalance, max_pipe_load, end, core)
            if best is None or choice < best:
                best = choice
        assert best is not None
        _, _, _, end, core = best
        candidate_edge_bytes = 0
        for pred in partition.preds:
            if scenario == "q1" or (communication_aware and core_of.get(pred) != core):
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
        # 后继前驱计数 -1，归零则加入 ready。
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
    cache_priority: bool | None = None,
    communication_aware: bool = True,
) -> list[list[int]]:
    """带关键路径粘性 + 缓存优先级的复杂图调度器。

    与 ``_schedule_partitions`` 的差异：ready 把 critical_set 放最前面、按路
    径序优先；每个候选分区选“粘性核”，放在非粘性核则加 ``_complex_split_penalty``
    惩罚。``cache_priority`` 与基础调度器同语义。
    """
    if num_cores < 1:
        raise ValueError("num_cores must be positive")
    if scenario not in {"q1", "q2", "q3"}:
        raise ValueError("scenario must be q1, q2, or q3")

    by_id = {p.id: p for p in partitions}
    topo = _partition_topology(partitions)
    topo_position = {pid: index for index, pid in enumerate(topo)}
    critical_index = {pid: index for index, pid in enumerate(critical_path)}
    critical_set = set(critical_path)
    # critical_predecessor: 关键路径上每个分区的“前一个”分区。
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
        if scenario == "q3" and _q3_reuse_priority(features, cache_priority)
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
        # 选粘性核：保护关键路径前驱所在核。
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
                elif communication_aware and core_of.get(pred) != core:
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
            ddr_lb = (estimated_ddr_bytes + candidate_edge_bytes) / bandwidth if communication_aware else 0
            objective_end = max(float(end), ddr_lb)
            # 放在非粘性核 → 加 split_penalty（仅 communication_aware 时）。
            if communication_aware and sticky_core is not None and core != sticky_core:
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
            if scenario == "q1" or (communication_aware and core_of.get(pred) != core):
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
    """为复杂图汇合处挑出一个值得保留的前驱核。

    优先保留关键路径前驱所在核；否则取共享字节数最大的前驱所在核；如果该
    分区只有 1 个前驱且无共享字节，则不强制粘性（返回 None）。
    """
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
    """复杂链路被拆开时的额外惩罚。

    取所有“已落核前驱”与当前分区之间最大的边字节数，按 60 bytes/cycle 折算
    传输时间，加上 scenario 固定等待（Q1=1000，Q2/Q3=500），再乘 scale。
    """
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
    """两个分区之间所有 source_op→target_op 边的最大字节数。"""
    source_ops = partitions[source].ops
    target_ops = partitions[target].ops
    return max(
        (_edge_size(features, u, v) for u in source_ops for v in target_ops if v in features.succs[u]),
        default=0,
    )


def _partition_cache_reuse(
    partitions: list[Partition], features: GraphFeatures
) -> dict[int, int]:
    """估计 Q3 缓存复用价值：被多个分区消费的张量 × 张量大小 × (复用次数)。

    一个张量被 N 个分区消费时，第二次之后的访问可以命中 L2，因此乘以
    (N-1)。结果仅作“优先级信号”，不是精确预算。
    """
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


def _q3_reuse_priority(features: GraphFeatures, force: bool | None = None) -> bool:
    """是否启用缓存优先级排序。

    默认仅对分支重图族（CNN_RESIDUAL / GATED_SIGMOID_MLP）启用，因为：
        * 这两类 motif 张量多路广播多，缓存收益高；
        * 其他图族单播为主，启用反而会把调度次序打乱得不偿失。

    ``force`` 不为 None 时直接返回 force，用于消融：
        * True  —— global_cache_priority；
        * False —— no_cache_priority。
    """
    from .graph_patterns import GraphPattern, classify_features

    if force is not None:
        return force
    return classify_features(features).pattern in {
        GraphPattern.CNN_RESIDUAL,
        GraphPattern.GATED_SIGMOID_MLP,
    }


def build_algorithm_plan(
    features: GraphFeatures,
    num_cores: int = 4,
    scenario: str = "q2",
    ablation: str | None = None,
) -> AlgorithmResult:
    """根据图族模式选择完整策略构建 Q3 方案。

    Q3 完整策略树在本文件内闭环，scenario 只用来切换 Q3 自己的成本模型。
    消融分支只在“no_cache_priority / global_cache_priority / no_communication”
    时重调度一次；其他分支沿用图族路由结果。
    """
    # Q3 的完整策略树在本文件内闭环，scenario 只用于选择 Q3 成本模型。
    from .graph_patterns import GraphPatternFamily, classify_features

    if ablation not in ABLATIONS:
        raise ValueError(f"unknown q3 ablation: {ablation!r}")
    graph_pattern = classify_features(features)
    # 第一步：按图族路由（与不带消融时一样）。
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
    # 把消融开关映射成调度参数：
    #   * communication_aware —— 仅 no_communication 关闭；
    #   * cache_priority      —— global_cache_priority→True；
    #                             no_cache_priority→False；
    #                             其他→None（走默认按图族判定）。
    communication_aware = ablation != "no_communication"
    cache_priority = True if ablation == "global_cache_priority" else (
        False if ablation == "no_cache_priority" else None
    )
    # 三种消融需要“重调度”一次（带新开关），让结果与开关一致。
    if ablation in {"no_cache_priority", "global_cache_priority", "no_communication"}:
        critical_path = _critical_partition_path(partitions, features)
        if graph_pattern.family.name in {"COMPLEX", "MIXED"} and critical_path:
            core_orders = _schedule_complex_partitions(
                partitions, features, num_cores, scenario, critical_path,
                split_penalty_scale=0.35 if graph_pattern.family.name == "MIXED" else 1.0,
                cache_priority=cache_priority,
                communication_aware=communication_aware,
            )
        else:
            core_orders = _schedule_partitions(
                partitions, features, num_cores, scenario,
                cache_priority=cache_priority,
                communication_aware=communication_aware,
            )
    # L2 开关：no_l2 关闭；其他变体保留 L2 复用本身。
    if ablation == "no_l2":
        strategy_diagnostics["l2_reuse"] = False
    else:
        strategy_diagnostics["l2_reuse"] = True
    if strategy_diagnostics.get("effective_core_count", num_cores) < num_cores:
        # 短副本限制会刻意留出尾部核空闲；全局迁移绝不能用小分区重新塞满它们，
        # 否则会破坏副本分组与 stage 交错。
        rebalance_diagnostics = {
            "enabled": False,
            "reason": "short_replicated_chain_core_cap",
            "moves": 0,
        }
    else:
        core_orders, rebalance_diagnostics = rebalance_core_orders(
            partitions, features, core_orders, scenario
        )
    strategy_diagnostics["global_rebalance"] = rebalance_diagnostics
    strategy_diagnostics["variant"] = ablation or "full"
    strategy_diagnostics["partition_count"] = len(partitions)
    node_to_subgraph = {
        str(op_id): partition.id
        for partition in partitions
        for op_id in partition.ops
    }
    critical_path = _critical_partition_path(partitions, features)
    owners = {pid: core for core, order in enumerate(core_orders) for pid in order}
    loads = [sum(partitions[pid].cycles for pid in order) for order in core_orders]
    mean_load = sum(loads) / max(1, len(loads))
    strategy_diagnostics.update({
        "critical_path_partition_ids": critical_path,
        "critical_path_length": len(critical_path),
        "critical_path_cycles": sum(partitions[pid].cycles for pid in critical_path),
        "critical_path_cross_core_edges": sum(
            owners.get(left) != owners.get(right) for left, right in zip(critical_path, critical_path[1:])
        ),
        "core_loads": loads,
        "load_cv": math.sqrt(sum((load - mean_load) ** 2 for load in loads) / len(loads)) / mean_load if mean_load else 0.0,
        "max_load_ratio": max(loads, default=0) / mean_load if mean_load else 0.0,
        "communication_aware": communication_aware,
        "cache_priority": cache_priority if cache_priority is not None else _q3_reuse_priority(features),
    })
    # 空核刻意保留在输出中；方案中带 q3_cache_mode，让评测器知道是否启用 L2。
    return AlgorithmResult(
        plan={
            "node_to_subgraph": node_to_subgraph,
            "core_schedules": core_orders,
            "q3_cache_mode": "disabled" if ablation == "no_l2" else "read_only_fifo",
        },
        diagnostics={
            "graph_pattern": graph_pattern.as_dict(),
            "algorithm": strategy_diagnostics,
        },
    )


def build_plan(
    graph: dict,
    num_cores: int = 4,
    features: GraphFeatures | None = None,
    ablation: str | None = None,
) -> AlgorithmResult:
    """对外入口显式固定 Q3 场景，避免误用 Q1/Q2 的入口配置。

    构建后会在 diagnostics 里补充 question / cache_aware / cache_model / variant
    元信息，便于评测器与 CSV 工具统一识别。
    """
    # 对外入口显式固定 Q3 场景，避免误用 Q1/Q2 的入口配置。
    if features is None:
        features = analyze_graph(graph)
    result = build_algorithm_plan(features, num_cores, scenario="q3", ablation=ablation)
    diagnostics = dict(result.diagnostics)
    algorithm = dict(diagnostics.get("algorithm", {}))
    algorithm.update({
        "question": "q3",
        "cache_aware": ablation not in {"no_l2", "no_cache_priority"},
        "cache_model": "read_only_fifo_copy_in",
        "variant": ablation or "full",
    })
    diagnostics["algorithm"] = algorithm
    return AlgorithmResult(plan=result.plan, diagnostics=diagnostics)

__all__ = ["build_plan"]
