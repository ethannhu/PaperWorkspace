# ============================================================
# 人工智能工具信息 | AI Tool Information
#   工具名称 (Tool Name)        : GLM-5.2
#   版本/型号 (Version/Model)   : GLM-5.2
#   开发机构/公司 (Developer)    : 智谱AI (Zhipu AI / zai-org)
#   版本颁布日期 (Release Date) : 2026-06-13
#   声明：本程序及代码是在人工智能工具辅助下完成的
# ============================================================
"""Q1 独立分区与调度算法。

Q1 将子图边界视为显式通信成本，优先合并连续计算并尽量保持关键路径的
同核亲和性。本文件包含 Q1 自己的分区、调度和入口实现，不引用 Q2/Q3。

Q1 的关键参数（来自 ``artifacts/data/config.txt``，不是调参结果）：
    * 同核等待 = 100 cycles；
    * 跨核固定等待 = 1000 cycles；
    * 跨核字节传输带宽 = 60 bytes/cycle，即传输时间 = ceil(bytes/60)。

策略树（按 ``GraphPattern`` 路由）：
    * WIDE_MATMUL_ADD / GATED_SIGMOID_MLP → 走 Q1 自有边界感知大块策略；
    * WIDE 家族其他 motif                  → 通信感知合并 + 列表调度；
    * COMPLEX 家族                         → 语义融合 + 关键路径粘性调度；
    * MIXED/NARROW                         → 默认语义分区 + 列表调度。

支持五组消融：no_semantic / no_adaptive / no_critical_path / no_rebalance。
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
ABLATIONS = {None, "no_semantic", "no_adaptive", "no_critical_path", "no_rebalance"}


def _semantic_plan(
    features: GraphFeatures,
    num_cores: int,
    scenario: str,
) -> tuple[list[Partition], list[list[int]], dict[str, Any]]:
    """默认完整策略：语义分块后再由本文件的列表调度器分到各核。"""
    # 先按语义合并算子，再由 Q1 的本地调度器分配到各个核。
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
            # 当前组已达到上限，开新组。
            groups.append(current)
            current = []
            current_ops = 0
            current_cycles = 0
        current.append(partition)
        current_ops += len(partition.ops)
        current_cycles += partition.cycles
    if current:
        groups.append(current)

    # 把每组内多个 Partition 累加成一个新 Partition：合并 ops / cycles /
    # pipe_cycles，rank_u 取组内最大值。再回填跨组 preds/succs。
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

    base = semantic_partition(features, max_ops=16, max_cycles=20000)
    operator_count = len(features.topo_order)
    if pattern == GraphPattern.WIDE_MATMUL_ADD:
        # 长、中等宽度的 ADD 主链需要更小块，调度器才能跨核流水线。但很宽的
        # 浅批则更适合大块以节省通信。
        layer_width = max(
            (
                sum(features.depth[node] == level for node in features.topo_order)
                for level in set(features.depth.values())
            ),
            default=0,
        )
        if operator_count < 5000:
            # 小图直接复用语义分块，避免额外的 coalesce。
            max_ops, max_cycles = 16, 20000
            motif = "semantic_wide_fallback"
        elif max(features.depth.values(), default=0) >= 16 and layer_width < 1000:
            # 中等深度 + 中等宽度 → 小块 + 轮转流水线。
            max_ops, max_cycles = 8, 6000
            motif = "matmul_add_pipelined_spine"
        else:
            # 很宽的浅图 → 大块节省 DMA。
            max_ops, max_cycles = 32, 18000
            motif = "matmul_add_fan_in"
    elif pattern == GraphPattern.GATED_SIGMOID_MLP:
        max_ops, max_cycles = 24, 18000
        motif = "gated_chain"
    else:
        max_ops, max_cycles = 24, 16000
        motif = "compute_activation_branch"
    if pattern == GraphPattern.SHALLOW_WIDE_COMPUTE_ACTIVATION and operator_count < 5000:
        # 浅宽且规模不大，直接退回纯语义分块。
        max_ops, max_cycles = 16, 20000
        motif = "semantic_wide_fallback"
    if motif == "semantic_wide_fallback":
        partitions = base
    else:
        partitions = _coalesce_topological_partitions(base, max_ops, max_cycles)
    if motif == "matmul_add_pipelined_spine":
        # 主链长 → 轮转散开，让相邻 stage 跨核重叠。
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
    critical_affinity: bool = True,
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
            # 副本独立 → 按 LPT（最大处理时间优先）打包到各核，按 stage 交错。
            core_orders = _schedule_replicated_components(
                partitions, num_cores
            )
            scheduler = "balanced_replicated_component_list"
        else:
            # 关键路径过长但不是副本 → 仍然只走普通调度，避免粘性加剧过载。
            core_orders = _schedule_partitions(partitions, features, num_cores, scenario)
            scheduler = "plain_list_after_complex_fusion"
    else:
        if critical_affinity:
            # 默认：关键路径粘性调度。
            core_orders = _schedule_complex_partitions(
                partitions, features, num_cores, scenario, critical_path
            )
            scheduler = "critical_path_sticky"
        else:
            # 消融：关闭关键路径粘性。
            core_orders = _schedule_partitions(partitions, features, num_cores, scenario)
            scheduler = "plain_list_without_critical_path"
    return partitions, core_orders, {
        "strategy": "complex_semantic_critical_path",
        "scheduler": scheduler,
        "partition_count": len(partitions),
        "critical_path_length": len(critical_path),
        "critical_path_cycles": critical_cycles,
        "max_layer_width": layer_width,
        "replicated_narrow_cnn": replicated_narrow_cnn,
        "critical_path_affinity": critical_affinity,
    }


def _mixed_plan(
    features: GraphFeatures,
    num_cores: int,
    scenario: str,
    critical_affinity: bool = True,
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
    # 宽/深图用更小块（10 ops / 12000 cycles），其余用稍大块（14 ops / 18000）。
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
    if critical_affinity:
        # 0.35 比默认 1.0 更柔和，避免过强的路径粘性把并行度也拖死。
        core_orders = _schedule_complex_partitions(
            partitions, features, num_cores, scenario, critical_path,
            split_penalty_scale=0.35,
        )
        scheduler = "critical_path_soft_sticky"
    else:
        core_orders = _schedule_partitions(partitions, features, num_cores, scenario)
        scheduler = "plain_list_without_critical_path"
    return partitions, core_orders, {
        "strategy": "mixed_semantic_fusion_critical_path",
        "scheduler": scheduler,
        "critical_path_affinity": critical_affinity,
        "partition_count": len(partitions),
        "critical_path_length": len(critical_path),
        "critical_path_cycles": critical_cycles,
        "max_ops": max_ops,
        "max_cycles": max_cycles,
        "singleton_repair": True,
    }


def _partition_topology(partitions: list[Partition]) -> list[int]:
    """对分区 DAG 跑一次拓扑排序，结果作为调度时的 tie-break。"""
    preds = {p.id: set(p.preds) for p in partitions}
    succs = {p.id: set(p.succs) for p in partitions}
    return _topological_order([p.id for p in partitions], preds, succs)


def _topology_only_partition(
    features: GraphFeatures, max_ops: int, max_cycles: int
) -> list[Partition]:
    """只按拓扑序与固定大小切块（A1 消融 no_semantic 专用）。

    不参考算子语义、不应用 motif 合并；纯按 ``features.topo_order`` 顺序扫描，
    当 ``max_ops`` 或 ``max_cycles`` 即将超限就开新分区。
    """
    groups: list[list[int]] = []
    current: list[int] = []
    cycles = 0
    for op_id in features.topo_order:
        op_cycles = int(features.op_by_id[op_id].get("cycles", 0))
        if current and (len(current) >= max_ops or cycles + op_cycles > max_cycles):
            groups.append(current)
            current, cycles = [], 0
        current.append(op_id)
        cycles += op_cycles
    if current:
        groups.append(current)

    owner = {op_id: pid for pid, ops in enumerate(groups) for op_id in ops}
    result: list[Partition] = []
    for pid, ops in enumerate(groups):
        # 同样统计每个分区的 pipe_cycles，方便后续调度器估管线负载。
        pipe_cycles: dict[str, int] = {}
        for op_id in ops:
            pipe = str(features.op_by_id[op_id].get("pipe", "UNKNOWN"))
            pipe_cycles[pipe] = pipe_cycles.get(pipe, 0) + int(
                features.op_by_id[op_id].get("cycles", 0)
            )
        result.append(Partition(
            id=pid,
            ops=ops,
            cycles=sum(int(features.op_by_id[op].get("cycles", 0)) for op in ops),
            rank_u=max((features.rank_u[op] for op in ops), default=0),
            pipe_cycles=pipe_cycles,
        ))
    for source in features.topo_order:
        for target in features.succs[source]:
            source_pid, target_pid = owner[source], owner[target]
            if source_pid != target_pid:
                result[source_pid].succs.add(target_pid)
                result[target_pid].preds.add(source_pid)
    return result


def _full_partition_reference(
    features: GraphFeatures, graph_pattern: Any
) -> list[Partition]:
    """跑一次 Full 策略，仅用结果粒度校准 A1 的拓扑切块大小。

    本函数返回的分区**不会**用于 A1 自身；只有它们的“数量”和“总 cycles”
    用来反推 A1 的固定切块上限，使 A1 与 Full 的颗粒度可比较，但不参考算子
    语义。
    """
    from .graph_patterns import GraphPattern, GraphPatternFamily

    if graph_pattern.pattern in {GraphPattern.GATED_SIGMOID_MLP, GraphPattern.WIDE_MATMUL_ADD}:
        partitions, _, _ = _legacy_q1_plan(features, graph_pattern.pattern, 1)
    elif graph_pattern.family == GraphPatternFamily.WIDE:
        partitions, _, _ = _wide_plan(features, 1, "q1", graph_pattern.pattern)
    elif graph_pattern.family == GraphPatternFamily.COMPLEX:
        partitions, _, _ = _complex_plan(features, 1, "q1", graph_pattern.pattern)
    else:
        partitions, _, _ = _semantic_plan(features, 1, "q1")
    return partitions


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

    # 1) 找连通分量：neighbours = preds ∪ succs，DFS 找到每个不连通组件。
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

    # 2) 每个组件内部按拓扑序排列；组件之间按 work 降序打包（即 LPT）。
    topo = _partition_topology(partitions)
    topo_index = {pid: index for index, pid in enumerate(topo)}
    ordered_components: list[tuple[int, list[int], int]] = []
    for component in components:
        ordered = sorted(component, key=topo_index.__getitem__)
        work = sum(by_id[pid].cycles for pid in ordered)
        ordered_components.append((work, ordered, min(ordered)))
    ordered_components.sort(key=lambda item: (-item[0], item[2]))

    # 3) LPT：每次把组件扔给当前负载最轻的核。
    assigned_components: list[list[list[int]]] = [
        [] for _ in range(num_cores)
    ]
    core_load = [0] * num_cores
    for work, component, _ in ordered_components:
        core = min(range(num_cores), key=lambda index: (core_load[index], index))
        assigned_components[core].append(component)
        core_load[core] += work

    # 4) 关键：同核内多副本按 stage 交错拼接，而不是直接连整条 branch。
    # 直接连接会让第一条 branch 整条先跑、第二条才能开始；交错拼接让 Q2/Q3
    # 同核张量复用得以生效（特别是当多个副本落在同核时）。
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
    """找出分区 DAG 上的一条重下游路径。

    算法：拓扑逆序计算每个分区的“包含边传输时间的下游关键路径长度”，再从
    分数最高的分区开始顺着 next_on_path 一路向下，得到关键路径。
    """
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
    # 起点：分数最大；tie-break 优先选 rank_u 大的，再选 id 小的。
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
    """关键路径优先的列表调度器（同构核）。

    调度流程：每个 ready 分区按 ``(rank_u ↓, cache_reuse ↓, topo_position ↑)``
    排序后取出。对每个核估出：
        1. 依赖就绪时间（含跨核延迟）；
        2. 计算结束时间；
        3. 管线负载不均衡度；
        4. DDR 字节下界。
    最终用 5 元组 ``(objective_end, pipe_imbalance, max_pipe_load, end, core)``
    作为目标：先比目标结束时间，再比管线不均衡、最大管线负载、真实 end、核
    id。目标是“选最可能减轻 makespan 与 DDR 压力的核”。
    """
    # 每次从就绪队列中选择关键路径最长的分区，并比较各核的预计结束时间。
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
        if scenario == "q3" and _q3_reuse_priority(features)
        else {}
    )

    while ready:
        # 排序：rank_u 大的优先；Q3 时 cache_reuse 大的优先；最后拓扑位置小的优先。
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
                    # scene-A：跨核与同核都要等待；同核 100，跨核 1000 + bytes/60。
                    edge_bytes = _partition_edge_size(partitions, features, pred, pid)
                    candidate_edge_bytes += edge_bytes
                    delay = (
                        same_core_wait if core_of.get(pred) == core else cross_core_wait
                    ) + math.ceil(edge_bytes / bandwidth)
                elif core_of.get(pred) != core:
                    # scene-B：只跨核才有等；固定 500 + bytes/60。
                    edge_bytes = _partition_edge_size(partitions, features, pred, pid)
                    candidate_edge_bytes += edge_bytes
                    delay = 500 + math.ceil(edge_bytes / bandwidth)
                dependency_ready = max(dependency_ready, finish[pred] + delay)
            start = max(core_time[core], dependency_ready)
            end = start + partition.cycles
            # Pipe 负载是评测器核内重叠执行的下界信号。保留保守“总 cycles end”
            # 估计，同时优先选管线压力更小的核。
            pipe_load = dict(core_pipe_load[core])
            for pipe, work in partition.pipe_cycles.items():
                pipe_load[pipe] = pipe_load.get(pipe, 0) + work
            max_pipe_load = max(pipe_load.values(), default=0)
            total_pipe_work = sum(pipe_load.values())
            avg_pipe_load = total_pipe_work / max(1, len(pipe_load))
            pipe_imbalance = max_pipe_load / max(1.0, avg_pipe_load)
            # DDR 下界：累计跨核字节 / 60。它无法藏在计算时间下面，所以目标
            # end 取 max(end, ddr_lb)。
            ddr_lb = (estimated_ddr_bytes + candidate_edge_bytes) / bandwidth
            objective_end = max(float(end), ddr_lb)
            choice = (objective_end, pipe_imbalance, max_pipe_load, end, core)
            if best is None or choice < best:
                best = choice
        assert best is not None
        _, _, _, end, core = best
        # 落定核数后，重算一次本次分区实际新增的跨核字节（与上面 for 循环语义
        # 一致，但 core 已确定），累计到 estimated_ddr_bytes。
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
        # 释放后继：减剩余前驱计数；为零则进入 ready。
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
    """带关键路径粘性的复杂图调度器。

    与 ``_schedule_partitions`` 的差异：
        * ready 排序里把 critical_set 上的分区放最前面，并按路径序优先；
        * 每个候选分区先经 ``_preferred_complex_core`` 选出“粘性核”（保护前驱
          所在核），若放在非粘性核则给目标 end 加一个 ``_complex_split_penalty``
          的惩罚；
        * 其余打分机制与基础调度器相同。
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
    # critical_predecessor: 关键路径上每个分区的“前一个”分区，用于选粘性核。
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
        # 排序：critical_set 优先；路径序优先；其余同基础调度器。
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
            # 放在非粘性核 → 加 split_penalty（按 scenario 取 1000 或 500 固定
            # 等待 + ceil(bytes/60) 传输）。惩罚 × ``split_penalty_scale`` 后
            # 加到目标 end 上。
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
    # 在所有已落核前驱中挑“边字节最大、rank_u 最大、id 最小”的作为代表。
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


def _q3_reuse_priority(features: GraphFeatures) -> bool:
    """仅对分支重图族开启缓存优先级排序（CNN/GATED）。"""
    from .graph_patterns import GraphPattern, classify_features

    return classify_features(features).pattern in {
        GraphPattern.CNN_RESIDUAL,
        GraphPattern.GATED_SIGMOID_MLP,
    }



def _q1_partitions(features: GraphFeatures, pattern: Any) -> tuple[list[Partition], dict[str, Any]]:
    """构建 Q1 较大的分区：Q1 对边界通信敏感，倾向用大块降低显式边界数。

    按图族选 ``max_ops``/``max_cycles``：
        * WIDE_MATMUL_ADD / SHALLOW_WIDE_COMPUTE_ACTIVATION → 24/32000（宽<1000）
          或 20/24000；WIDE_MATMUL_ADD 在 depth≥16 且 width<1000 时使用 14/14000
          以提升流水线机会；
        * CNN_RESIDUAL → 宽 CNN 28/60000，窄 CNN 24/45000；
        * GATED_SIGMOID_MLP → 28/42000；
        * 其他 → 宽图 22/30000，窄图 28/42000。

    分区后再调用一次 ``_coalesce_topological_partitions`` 去掉一些剩余的串行
    边界（CNN 家族不加 coalesce，避免破坏已修复的 singleton）。
    """
    # Q1 对边界通信更敏感，因此使用较大的分区降低显式边界数量。
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
    # 语义合并对汇合点刻意保守。再用一次有界拓扑合并去掉一些剩余的串行边界。
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


def _legacy_q1_plan(
    features: GraphFeatures,
    pattern: Any,
    num_cores: int,
    critical_affinity: bool = True,
) -> tuple[list[Partition], list[list[int]], dict[str, Any]]:
    """为 Q1 自有边界策略收益更高的图族构建原始 Q1 方案。

    适用 WIDE_MATMUL_ADD、GATED_SIGMOID_MLP 等用大块 + 边界感知更划算的
    motif。COMPLEX 家族 + 有关键路径时走粘性调度，否则走 affinity 列表。
    """
    partitions, diagnostics = _q1_partitions(features, pattern)
    critical_path = _critical_partition_path(partitions, features)
    from .graph_patterns import GraphPatternFamily, classify_features

    graph_pattern = classify_features(features)
    if graph_pattern.family == GraphPatternFamily.COMPLEX and critical_path and critical_affinity:
        core_orders = _schedule_complex_partitions(
            partitions, features, num_cores, "q1", critical_path
        )
        diagnostics["scheduler"] = "q1_critical_path_sticky"
    else:
        core_orders = _schedule_partitions(partitions, features, num_cores, "q1")
        diagnostics["scheduler"] = "q1_affinity_list"
    diagnostics["critical_path_length"] = len(critical_path)
    diagnostics["critical_path_affinity"] = critical_affinity
    return partitions, core_orders, diagnostics


def build_algorithm_plan(
    features: GraphFeatures, num_cores: int = 4, ablation: str | None = None
) -> AlgorithmResult:
    """构建 Q1 计划；路由、分区和调度均由本文件独立完成。

    流程：
        1. 根据 ``ablation`` 决定消融开关；
        2. 用 classify_features 路由到对应图族策略；
        3. A1/A2 消融 (no_semantic / no_adaptive) 跳过图族路由，走专门分支；
        4. 调度完成后（除非是 no_rebalance）跑再平衡迁移；
        5. 计算关键路径、负载 CV 等诊断字段。

    所有核数都使用同一套图族策略，保证单核基线与多核结果可比较。
    """
    # 构建路由、分区和调度均由本文件独立完成；路由后不再调用 Q2/Q3 的入口。
    from .graph_patterns import GraphPatternFamily, classify_features

    if num_cores < 1:
        raise ValueError("num_cores must be positive")
    if ablation not in ABLATIONS:
        raise ValueError(f"unknown q1 ablation: {ablation!r}")
    graph_pattern = classify_features(features)
    from .graph_patterns import GraphPattern

    # 所有核数使用同一套图族策略，保证单核基线和多核结果可比较。
    critical_affinity = ablation != "no_critical_path"
    if ablation in {"no_semantic", "no_adaptive"}:
        from .semantic_partition import semantic_partition

        if ablation == "no_semantic":
            # A1：保留 Q1 的大小上限，但用纯拓扑顺序切块替换语义合并。
            # 用 Full 的分区数校准 A1 的固定上限，使两者颗粒度可比。
            reference_partitions = _full_partition_reference(features, graph_pattern)
            target_count = max(1, len(reference_partitions))
            total_cycles = sum(
                int(features.op_by_id[op_id].get("cycles", 0))
                for op_id in features.topo_order
            )
            partitions = _topology_only_partition(
                features,
                max(1, math.ceil(len(features.topo_order) / target_count)),
                max(1, math.ceil(total_cycles / target_count)),
            )
            strategy = "topology_only_chunking"
        else:
            # A2：所有图统一切成 24 ops / 32000 cycles，禁用按图族自适应粒度。
            partitions = semantic_partition(
                features, max_ops=24, max_cycles=32000,
                enable_singleton_repair=True,
            )
            strategy = "fixed_granularity_semantic"
        critical_path = _critical_partition_path(partitions, features)
        if graph_pattern.family.name == "COMPLEX" and critical_path and critical_affinity:
            core_orders = _schedule_complex_partitions(
                partitions, features, num_cores, "q1", critical_path
            )
            scheduler = "critical_path_sticky"
        else:
            core_orders = _schedule_partitions(partitions, features, num_cores, "q1")
            scheduler = "plain_list"
        diagnostics = {
            "strategy": strategy,
            "scheduler": scheduler,
            "partition_count": len(partitions),
            "max_ops": 24 if ablation == "no_adaptive" else math.ceil(len(features.topo_order) / max(1, len(_full_partition_reference(features, graph_pattern)))),
            "max_cycles": 32000 if ablation == "no_adaptive" else math.ceil(sum(int(features.op_by_id[op].get("cycles", 0)) for op in features.topo_order) / max(1, len(_full_partition_reference(features, graph_pattern)))),
            "critical_path_affinity": critical_affinity,
            "critical_path_length": len(critical_path),
        }
    elif graph_pattern.pattern in {
        GraphPattern.GATED_SIGMOID_MLP,
        GraphPattern.WIDE_MATMUL_ADD,
    }:
        # 这两个 motif 在 Q1 自有大块 + 边界感知下表现更好。
        partitions, core_orders, diagnostics = _legacy_q1_plan(
            features, graph_pattern.pattern, num_cores, critical_affinity
        )
    elif graph_pattern.family == GraphPatternFamily.WIDE:
        partitions, core_orders, diagnostics = _wide_plan(
            features, num_cores, "q1", graph_pattern.pattern
        )
    elif graph_pattern.family == GraphPatternFamily.COMPLEX:
        partitions, core_orders, diagnostics = _complex_plan(
            features, num_cores, "q1", graph_pattern.pattern, critical_affinity
        )
    elif graph_pattern.family == GraphPatternFamily.MIXED:
        partitions, core_orders, diagnostics = _semantic_plan(
            features, num_cores, "q1"
        )
    else:
        # NARROW 家族与其他未命中分支都走默认语义分区。
        partitions, core_orders, diagnostics = _semantic_plan(
            features, num_cores, "q1"
        )

    if ablation == "no_rebalance":
        rebalance_diagnostics = {"enabled": False, "moves": 0}
    else:
        core_orders, rebalance_diagnostics = rebalance_core_orders(
            partitions, features, core_orders, "q1"
        )
    diagnostics["global_rebalance"] = rebalance_diagnostics

    node_to_subgraph = {
        str(op_id): partition.id
        for partition in partitions
        for op_id in partition.ops
    }
    diagnostics["graph_pattern"] = graph_pattern.as_dict()
    diagnostics["variant"] = ablation or "full"
    diagnostics["partition_count"] = len(partitions)
    # 重新计算关键路径与跨核边数，作为诊断字段（与调度结果一致）。
    cp = _critical_partition_path(partitions, features)
    owners = {pid: core for core, order in enumerate(core_orders) for pid in order}
    diagnostics["critical_path_partition_ids"] = cp
    diagnostics["critical_path_length"] = len(cp)
    diagnostics["critical_path_cycles"] = sum(partitions[pid].cycles for pid in cp)
    diagnostics["critical_path_cross_core_edges"] = sum(
        owners.get(left) != owners.get(right)
        for left, right in zip(cp, cp[1:])
    )
    loads = [sum(partitions[pid].cycles for pid in order) for order in core_orders]
    mean_load = sum(loads) / max(1, len(loads))
    diagnostics["core_loads"] = loads
    # 负载变异系数 CV：标准差/均值，度量负载均衡程度。
    diagnostics["load_cv"] = (
        math.sqrt(sum((load - mean_load) ** 2 for load in loads) / len(loads)) / mean_load
        if mean_load else 0.0
    )
    diagnostics["max_load_ratio"] = max(loads, default=0) / mean_load if mean_load else 0.0
    return AlgorithmResult(
        plan={"node_to_subgraph": node_to_subgraph, "core_schedules": core_orders},
        diagnostics={"algorithm": diagnostics},
    )


def build_plan(
    graph: dict[str, Any],
    num_cores: int = 4,
    features: GraphFeatures | None = None,
    ablation: str | None = None,
) -> AlgorithmResult:
    """Q1 框架入口。

    评测框架只调用这个入口；特征可复用，但算法实现始终属于 Q1 文件。
    """
    # 评测框架只调用这个入口；特征可复用，但算法实现始终属于 Q1 文件。
    if features is None:
        features = analyze_graph(graph)
    return build_algorithm_plan(features, num_cores, ablation)


__all__ = ["build_plan", "build_algorithm_plan"]
