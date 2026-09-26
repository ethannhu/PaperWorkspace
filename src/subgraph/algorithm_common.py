# ============================================================
# 人工智能工具信息 | AI Tool Information
#   工具名称 (Tool Name)        : GLM-5.2
#   版本/型号 (Version/Model)   : GLM-5.2
#   开发机构/公司 (Developer)    : 智谱AI (Zhipu AI / zai-org)
#   版本颁布日期 (Release Date) : 2026-06-13
#   声明：本程序及代码是在人工智能工具辅助下完成的
# ============================================================
"""Q1/Q2/Q3 三题算法共享的图分析类型与基础工具。

框架刻意停在比赛输出边界处：它不创建 COPY、不模拟 NPU，这两件事由官方
评测器完成。实现有意保持朴素：
    * Kahn 拓扑排序用于图分析；
    * 关键路径优先的列表调度用于同构核集群。

图族策略同时负责分区与调度。当前所有策略都使用语义分区器 + 同一个列表
调度器，但框架把每个图族分支视作一个完整的算法，方便后续替换。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from .interfaces import AlgorithmResult


# COPY 是评测器在子图边界自动插入的同步算子，用户方案不能直接出现。
COPY_TYPES = {"COPY_IN", "COPY_OUT"}
# “锚点”算子：MATMUL/CONV/REDUCE 通常是分区的主干，融合策略围绕它们展开。
ANCHOR_TYPES = {"MATMUL", "CONV", "REDUCE"}
# ELEMENTWISE 类：开销小、可串接到任何前驱后做轻量后处理。
ELEMENTWISE_TYPES = {
    "ADD", "MUL", "SUB", "DIV", "RELU", "SIGMOID", "EXP", "SQRT", "NEG"
}


@dataclass
class GraphFeatures:
    """基线算法消费的少量图特征。

    所有 dict 的键都是 op_id。``preds``/``succs`` 是 **COPY 收缩后** 的算子
    DAG，对应 ``edge_sizes`` 也只保留跨非 COPY 算子边的张量字节数估计。
    """

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
    """轻量分区对象，供所有调度器消费。

    ``pipe_cycles`` 记录每个 pipeline 上的 cycle 总和，调度器用它在 ddr 下界
    之外再做一次“管线不均衡”惩罚。``rank_u`` 取分区内任意算子的最大上秩，
    作为关键路径优先级的代理。
    """

    id: int
    ops: list[int]
    cycles: int
    rank_u: int
    pipe_cycles: dict[str, int] = field(default_factory=dict)
    preds: set[int] = field(default_factory=set)
    succs: set[int] = field(default_factory=set)


def classify_op(op_type: str) -> str:
    """把单个算子类型映射到统一语义标签。

    返回值之一：DENSE_COMPUTE / REDUCTION / ELEMENTWISE / COMMUNICATION / OTHER。
    """
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
    """把 op→tensor→op 和直接 op→op 边统一转换成算子 DAG。

    同时记录每条算子边的累计张量字节数。``producers``/``consumers`` 用来桥接
    tensor 中转的隐式依赖：A 写出 tensor T、B 读 T，则在 A→B 之间补一条边。
    """
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
                # 一个 producer/consumer 对可能由多个 tensor 连接。通信字节数
                # 取这些 tensor 的总和，而不是只用最大的那条 tensor。
                edge_sizes[(source, target)] = edge_sizes.get((source, target), 0) + size
    return preds, succs, edge_sizes


def _topological_order(nodes: Iterable[int], preds: dict[int, set[int]], succs: dict[int, set[int]]) -> list[int]:
    """Kahn 拓扑排序；同一批就绪节点按 id 排序以稳定输出。

    若检测结果不等于输入节点数，说明存在环，直接抛错。比赛数据保证是 DAG，
    但本函数同时被“收缩后图”反复调用，环是早期 BUG 的高发点。
    """
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
    """把原始 JSON 图转成 ``GraphFeatures``。

    主要工作：
        1. 构造 op DAG（含 tensor 隐式依赖）；
        2. 把 COPY 节点“收缩”掉，使得非 COPY 算子之间形成等价的直接依赖；
        3. 在复制后的图上跑拓扑排序，统计 depth 与关键路径上秩 rank_u；
        4. 重新整理每个非 COPY 算子的输入/输出 tensor，供缓存复用与边界估计使用。
    """
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

    # COPY 节点是边界，不属于用户分区。在算 eligible 算子的拓扑之前先把它们
    # 收缩掉：通过 COPY 节点中转的依赖，被替换成“上游 eligible → 下游 eligible”
    # 的直接边。
    contracted_preds = {node: set() for node in eligible}
    contracted_succs = {node: set() for node in eligible}
    eligible_set = set(eligible)
    contracted_edge_sizes: dict[tuple[int, int], int] = {}
    for source in eligible:
        # 对每个 COPY 节点保留见到过的最大边界张量估计。COPY 收缩可能暴露
        # 多条路径，但只有当一个新路径携带的字节数更大时，重访该节点才有意义。
        stack = [(target, edge_sizes.get((source, target), 0)) for target in succs[source]]
        seen_weight: dict[int, int] = {}
        while stack:
            target, path_bytes = stack.pop()
            if target in eligible_set:
                if target != source:
                    contracted_succs[source].add(target)
                    contracted_preds[target].add(source)
                    key = (source, target)
                    # 收缩图上的边权取所有路径中携带字节的最大值：评测器是按
                    # “每个目标 task 一次 DMA”计费的，取最大值更接近真实成本。
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
    # depth = 拓扑层数；rank_u = 从该节点到下游汇点的最长“cycle 加权和”，
    # 两者都在收缩后的算子 DAG 上计算。
    depth: dict[int, int] = {}
    rank_u: dict[int, int] = {}
    for node in topo:
        depth[node] = 0 if not contracted_preds[node] else 1 + max(depth[p] for p in contracted_preds[node])
    for node in reversed(topo):
        cycles = int(op_by_id[node].get("cycles", 0))
        rank_u[node] = cycles if not contracted_succs[node] else cycles + max(
            rank_u[s] for s in contracted_succs[node]
        )

    # 返回收缩后的非 COPY 算子依赖。原始 edge_size 查表对直接边仍然有用；
    # tensor 边在下面重新整理出来。
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
    """查询 source→target 之间的张量字节数。

    缺失元数据即视为没有已知张量传输；之前曾用“全图最大 tensor”兜底，但
    在 COPY 收缩后会把无关依赖也放大成大流量，故现在直接返回 0。
    """
    return features.edge_sizes.get((source, target), 0)


def _partition_edge_size(
    partitions: list[Partition],
    features: GraphFeatures,
    source: int,
    target: int,
) -> int:
    """返回两个分区之间最大的已知张量边字节数。

    评测器是按“每对 source_op→target_op 边”计 DMA 的；取最大值是对最坏边界
    成本的保守估计。rebalance 与调度器都使用这个函数估算跨核通信。
    """
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
    """用有界“整分区迁移”继续优化已完成的调度结果。

    正常调度器是贪心的：对每个就绪分区独立选核。本步把它的结果当作起点，
    只在“最重核 → 最轻核”之间搜索迁移。每次试探都会 replay 一次完整依赖
    调度，仅当 ``makespan + 0.1 × cross_bytes/60`` 的标量评分严格下降时才接受
    迁移。每轮结束后都按分区的拓扑序重建核内顺序，因此移动分区不会引入同核
    依赖环。

    注意：``replay`` 中的 100 / 1000 / 500 / ceil(bytes/60) 这些常量都不是
    调参结果，而是 ``artifacts/data/config.txt`` 中评测器的固定值。
    """
    if not core_orders or not partitions:
        return core_orders, {"enabled": True, "moves": 0}
    if scenario not in {"q1", "q2", "q3"}:
        raise ValueError("scenario must be q1, q2, or q3")

    by_id = {partition.id: partition for partition in partitions}
    # 用分区 DAG 自身的拓扑序重建核内顺序，避免迁移引入同核依赖环。
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
    # 对残缺/空方案做防御性补全；正常方案每个分区都恰好出现一次。
    for pid in topo:
        owner.setdefault(pid, min(range(len(core_orders)), key=lambda c: c))

    def rebuild(current_owner: dict[int, int]) -> list[list[int]]:
        # 按分区拓扑序把分区塞回各核，保证同核内顺序合法。
        orders = [[] for _ in core_orders]
        for pid in topo:
            orders[current_owner[pid]].append(pid)
        return orders

    def replay(current_owner: dict[int, int]) -> tuple[float, int, int, float]:
        """重放一次完整依赖调度，返回 (score, edge_bytes, max_load, makespan)。"""
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
                # scene-A：跨核与同核都有固定等待，跨核额外付 1000 + bytes/60；
                # scene-B/Q2/Q3：仅跨核需要等待，固定 500 + bytes/60，同核为 0。
                if scenario == "q1":
                    delay = (100 if not cross else 1000) + math.ceil(bytes_ / 60)
                else:
                    delay = (500 + math.ceil(bytes_ / 60)) if cross else 0
                ready = max(ready, finish[pred] + delay)
            previous_finish = finish.get(previous[pid], 0)
            # scene-A 中即使前序 task 与当前 task 没有数据依赖，下一个 task 也
            # 必须等当前核的同核等待周期才能释放。这是排队约束，不是边成本。
            if scenario == "q1" and previous[pid] != -1:
                previous_finish += 100
            finish[pid] = max(ready, previous_finish) + by_id[pid].cycles
        loads = [
            sum(by_id[pid].cycles for pid in order)
            for order in orders
        ]
        makespan = max(finish.values(), default=0)
        max_load = max(loads, default=0)
        # 通信只是次要项：计算/依赖时间仍占主导；当迁移在时间上等价时，惩罚
        # 大字节传输，避免为追求“通信略少”而胡乱搬迁。
        score = makespan + 0.10 * edge_bytes / 60.0
        return score, edge_bytes, max_load, makespan

    initial = replay(owner)
    current = initial
    moves = 0
    rounds = 0
    # 每轮最多评估 24 个候选迁移，避免在最重核很大时退化成 O(N²) 暴力搜索。
    candidate_limit = 24
    while rounds < max_rounds:
        rounds += 1
        orders = rebuild(owner)
        loads = [
            sum(by_id[pid].cycles for pid in order)
            for order in orders
        ]
        # source = 当前负载最重的核；targets = 其余核按负载从轻到重排序。
        source = max(range(len(orders)), key=lambda core: (loads[core], -core))
        targets = sorted(
            (core for core in range(len(orders)) if core != source),
            key=lambda core: (loads[core], core),
        )
        if not targets or loads[source] <= loads[targets[0]]:
            break
        # 优先迁移 cycle 大、拓扑位置靠后的分区，这样最可能显著降低 makespan。
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
                # 1e-9 防止浮点噪声触发无意义的“改进”。
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
    """输出与每份方案配套的图模式诊断字典。"""
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
