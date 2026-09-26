# ============================================================
# 人工智能工具信息 | AI Tool Information
#   工具名称 (Tool Name)        : GLM-5.2
#   版本/型号 (Version/Model)   : GLM-5.2
#   开发机构/公司 (Developer)    : 智谱AI (Zhipu AI / zai-org)
#   版本颁布日期 (Release Date) : 2026-06-13
#   声明：本程序及代码是在人工智能工具辅助下完成的
# ============================================================
"""基于语义与硬件感知的图分区器。

分区器遵循 ``docs/partition.md`` 的设计：

1. 用一组小型语义特征向量标注每个节点；
2. 从兼容的 motif 中组成局部 stage 块；
3. 当“通信 + 局部性收益”超过“并行度损失”时贪心合并块，再重建分区 DAG。

它刻意只返回当前图族策略使用的轻量 ``Partition`` 对象。打分公式中的常量
都是基于 ``artifacts/data`` 标定得到的，目的是在同一组阈值下覆盖所有 case：
    * communication_saved —— 用 ``log1p(transfer_cycles / 16)`` 把字节数压
      到与语义项同量级，保留 4KiB→32KiB 的差别但不让大 tensor 饱和；
    * locality_gain      —— 1.5；张量直接复用时给的小奖励；
    * semantic_affinity  —— 0~4.0；按角色组合查表给出；
    * resource_diversity —— +0.8/-0.4；跨 pipe 是否是显式 stage 边；
    * parallelism_loss   —— 2.5 × Bool(出度/入度>1)；保并行的硬惩罚；
    * boundary_penalty   —— 两端 boundary_score 之和（fan-in/out 越大越大）；
    * serial_penalty     —— 0.5；两端都是 COMPUTE 时的小串行惩罚。
"""

from __future__ import annotations

import heapq
import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .algorithm_common import GraphFeatures, Partition


# 五类语义角色常量集合。merge_score 通过角色组合查亲和力与是否允许融合。
COMPUTE_ROLES = {"DENSE_COMPUTE", "COMPUTE"}
REDUCTION_ROLES = {"REDUCTION"}
ELEMENTWISE_ROLES = {"ELEMENTWISE"}
COMMUNICATION_ROLES = {"COMMUNICATION"}
MEMORY_ROLES = {"MEMORY"}

# 以下是 ``artifacts/data/config.txt`` 中评测器的固定参数。打分公式的输入是
# “评测器认定的传输时间”而非裸字节数，所以这两个常数把字节数换算到与语义
# 项同量级。
_EVALUATOR_BANDWIDTH_BYTES_PER_CYCLE = 60
_COMMUNICATION_REFERENCE_CYCLES = 16


@dataclass(frozen=True)
class NodeFeature:
    """每个算子的语义特征向量。

    ``criticality`` 取 rank_u（关键路径长度），``boundary_score`` 用入度/出度
    的偏离 1 部分加权得到，是“合并此处会破坏多少并行度”的代理量。
    """

    id: int
    op: str
    pipe: str
    cycles: int
    role: str
    indegree: int
    outdegree: int
    input_tensors: tuple[int, ...]
    output_tensors: tuple[int, ...]
    criticality: float
    boundary_score: float


@dataclass
class SemanticBlock:
    """一组连续算子合并出的语义块（merge 的中间结构）。"""

    id: int
    nodes: list[int]
    semantic_type: str
    cycles: int
    pipe_cycles: dict[str, int] = field(default_factory=dict)
    input_tensors: set[int] = field(default_factory=set)
    output_tensors: set[int] = field(default_factory=set)
    internal_comm: float = 0.0
    external_comm: float = 0.0
    buffer_pressure: float = 0.0
    pipelineability: float = 0.0
    predecessors: set[int] = field(default_factory=set)
    successors: set[int] = field(default_factory=set)


def _role(op: str, current: str) -> str:
    """把框架的旧标签映射到设计文档中的语义角色。"""
    if op in {"ALLOC", "FREE"}:
        return "MEMORY"
    if op in {"COPY_IN", "COPY_OUT", "MOVE"} or current == "COMMUNICATION":
        return "COMMUNICATION"
    if current == "DENSE_COMPUTE":
        return "COMPUTE"
    if current == "REDUCTION":
        return "REDUCTION"
    if current == "ELEMENTWISE":
        return "ELEMENTWISE"
    return "OTHER"


def extract_features(features: GraphFeatures) -> dict[int, NodeFeature]:
    """为每个节点构造供 matcher/scorer 使用的语义特征向量。"""
    result: dict[int, NodeFeature] = {}
    for node in features.topo_order:
        op_data = features.op_by_id[node]
        op = str(op_data.get("op", ""))
        pipe = str(op_data.get("pipe", "UNKNOWN"))
        cycles = int(op_data.get("cycles", 0))
        indegree = len(features.preds[node])
        outdegree = len(features.succs[node])
        # 后继中 pipe 与当前节点不同的数量，用于度量“资源切换”。
        resource_switch = sum(
            1 for successor in features.succs[node]
            if str(features.op_by_id[successor].get("pipe", "UNKNOWN")) != pipe
        )
        # fan-in/out 是比 pipe 切换更强劲的“合并会破坏并行”信号：分支和汇合
        # 几乎总是意味着此处不应融合。因此权重比 resource_switch 大 4 倍。
        boundary_score = (
            2.0 * max(0, indegree - 1)
            + 2.0 * max(0, outdegree - 1)
            + 0.5 * resource_switch
        )
        result[node] = NodeFeature(
            id=node,
            op=op,
            pipe=pipe,
            cycles=cycles,
            role=_role(op, features.semantic[node]),
            indegree=indegree,
            outdegree=outdegree,
            input_tensors=tuple(sorted(features.input_tensors.get(node, set()))),
            output_tensors=tuple(sorted(features.output_tensors.get(node, set()))),
            criticality=float(features.rank_u[node]),
            boundary_score=boundary_score,
        )
    return result


def _initial_block(feature: NodeFeature) -> SemanticBlock:
    """把单个节点包装成初始的 singleton SemanticBlock。"""
    return SemanticBlock(
        id=feature.id,
        nodes=[feature.id],
        semantic_type=feature.role,
        cycles=feature.cycles,
        pipe_cycles={feature.pipe: feature.cycles},
        input_tensors=set(feature.input_tensors),
        output_tensors=set(feature.output_tensors),
        pipelineability=1.0,
    )


def _motif_type(left: NodeFeature, right: NodeFeature) -> str:
    """根据左右两端角色返回 motif 名称，仅用于 diagnostics 与可读性。"""
    roles = {left.role, right.role}
    if left.role == "COMPUTE" and right.role == "ELEMENTWISE":
        return "ComputeFusionBlock"
    if left.role == "REDUCTION" and right.role in {"ELEMENTWISE", "REDUCTION"}:
        return "ReductionStage"
    if left.role == "COMMUNICATION" or right.role == "COMMUNICATION":
        return "CommunicationComputeStage"
    if roles == {"COMPUTE"} and left.pipe == right.pipe:
        return "ComputeTile"
    if roles == {"ELEMENTWISE"}:
        return "ElementwiseChain"
    return "MixedStage"


def _semantic_affinity(left: NodeFeature, right: NodeFeature) -> float:
    """两节点之间的语义亲和度（0~4.0）。

    * COMPUTE→ELEMENTWISE 是经典算子+激活融合，给最高分 4.0；
    * REDUCTION→ELEMENTWISE/REDUCTION 是 reduce 后处理，2.5；
    * ELEMENTWISE→ELEMENTWISE 是链式后处理，2.5；
    * 同 pipe 不同角色，1.0；
    * PIPE_MTE2/3 ↔ PIPE_M 是硬件约定的 stage 切换，1.5；
    * 否则 0。
    """
    if left.role == "COMPUTE" and right.role == "ELEMENTWISE":
        return 4.0
    if left.role == right.role == "ELEMENTWISE":
        return 2.5
    if left.role == "REDUCTION" and right.role in {"ELEMENTWISE", "REDUCTION"}:
        return 2.5
    if left.pipe == right.pipe:
        return 1.0
    if {left.pipe, right.pipe} == {"PIPE_MTE2", "PIPE_M"}:
        return 1.5
    return 0.0


def _explicit_stage_pair(left: NodeFeature, right: NodeFeature) -> bool:
    """判断一条跨资源边是不是“设计上有意的 stage 边界”。

    跨 pipe 默认不允许合并（避免破坏硬件流水线）；本函数列出几种白名单
    组合，例外允许合并。最后一行保留 MTE→Cube→M 等硬件导向 motif，适用于
    把 MOVE/COPY 类 stage 暴露成普通算子的图。
    """
    if left.role == "COMPUTE" and right.role == "ELEMENTWISE":
        return True
    if left.role == "REDUCTION" and right.role in {"ELEMENTWISE", "REDUCTION"}:
        return True
    if left.role == "COMMUNICATION" and right.role == "COMPUTE":
        return True
    return (
        left.pipe in {"PIPE_MTE2", "PIPE_MTE3"} and right.pipe == "PIPE_M"
    ) or (
        left.pipe == "PIPE_M" and right.pipe in {"PIPE_MTE2", "PIPE_MTE3"}
    )


def _merge_score(
    left: SemanticBlock,
    right: SemanticBlock,
    features: dict[int, NodeFeature],
    edge_bytes: int,
    max_ops: int,
    max_cycles: int,
) -> float:
    """给一对相邻 block 的合并打分（详细分解见模块 docstring）。"""
    if len(left.nodes) + len(right.nodes) > max_ops:
        return float("-inf")
    if left.cycles + right.cycles > max_cycles:
        return float("-inf")
    tail = features[left.nodes[-1]]
    head = features[right.nodes[0]]
    # 一个 ELEMENTWISE producer 通常会喂给多个独立 compute tile。把它融进
    # 第一个 tile 会把 producer 与 cube stage 串行化，并把该 tile 推到 task
    # 优先级队列尾部。直接禁掉。
    if tail.role == "ELEMENTWISE" and head.role == "COMPUTE":
        return float("-inf")
    if tail.pipe != head.pipe and not _explicit_stage_pair(tail, head):
        return float("-inf")
    # 用“节省的 DMA 时间”给边界定价。直接用字节会让 6KiB 以上的 tensor 全部
    # 饱和到同一个奖励，对数项保留 4KiB→32KiB 的差别（69→547 传输周期）。
    communication_saved = _communication_saved(edge_bytes)
    locality_gain = 1.5 if left.output_tensors & right.input_tensors else 0.0
    resource_diversity = 0.8 if tail.pipe != head.pipe else -0.4
    parallelism_loss = 2.5 * (tail.outdegree > 1 or head.indegree > 1)
    boundary_penalty = tail.boundary_score + head.boundary_score
    serial_penalty = 0.5 if tail.role == head.role == "COMPUTE" else 0.0
    return (
        communication_saved
        + locality_gain
        + _semantic_affinity(tail, head)
        + resource_diversity
        - parallelism_loss
        - boundary_penalty
        - serial_penalty
    )


def _merge(left: SemanticBlock, right: SemanticBlock, motif: str) -> None:
    """把 right 并入 left（原地修改 left）。"""
    left.nodes.extend(right.nodes)
    left.semantic_type = motif if left.semantic_type != motif else left.semantic_type
    left.cycles += right.cycles
    for pipe, cycles in right.pipe_cycles.items():
        left.pipe_cycles[pipe] = left.pipe_cycles.get(pipe, 0) + cycles
    left.input_tensors |= right.input_tensors
    left.output_tensors |= right.output_tensors
    left.internal_comm += right.internal_comm
    left.buffer_pressure += right.buffer_pressure
    left.pipelineability = max(left.pipelineability, right.pipelineability)


def _owner_map(blocks: dict[int, SemanticBlock]) -> dict[int, int]:
    """node_id → 当前所属 block_id 的映射。"""
    return {node: block.id for block in blocks.values() for node in block.nodes}


def _edges_between(
    features: GraphFeatures,
    left: SemanticBlock,
    right: SemanticBlock,
) -> list[tuple[int, int]]:
    """枚举 left→right 之间所有直接依赖边。"""
    right_nodes = set(right.nodes)
    return [
        (source, target)
        for source in left.nodes
        for target in features.succs[source]
        if target in right_nodes
    ]


def _crossing_bytes(
    features: GraphFeatures,
    left: SemanticBlock,
    right: SemanticBlock,
    edge_pairs: list[tuple[int, int]] | None = None,
) -> int:
    """返回一对 block 边界处只 DMA 一次的张量字节总数。

    一个 tensor 可能被 right 中多个算子消费。如果按算子边求和，会把这条
    DMA 重复计；评测器按“每个目标 task 一次 DMA”计费，所以这里优先用 tensor
    集合求和。COPY 收缩可能掩盖 tensor 身份，此时退回到收缩边的字节数估计。
    """
    produced = set().union(
        *(features.output_tensors.get(node, set()) for node in left.nodes)
    )
    consumed = set().union(
        *(features.input_tensors.get(node, set()) for node in right.nodes)
    )
    shared_tensors = produced & consumed
    if shared_tensors:
        return sum(
            int(features.tensor_by_id[tensor_id].get("size", 0))
            for tensor_id in shared_tensors
        )
    if edge_pairs is None:
        edge_pairs = _edges_between(features, left, right)
    return sum(features.edge_sizes.get(edge, 0) for edge in edge_pairs)


def _communication_saved(edge_bytes: int) -> float:
    """把“避免的传输时间”归一化到 0~4 的同量级，便于和语义项相加。

    ``ceil(bytes / 60)`` 是官方评测器传输模型中的字节部分。固定等待时间被
    故意忽略：它是否生效取决于后续核放置与 Q1/Q2 场景，而语义分区阶段并不知道
    这些。``_COMMUNICATION_REFERENCE_CYCLES = 16`` 是评分标尺参考，不是硬件
    参数；它把这个奖励保持在和 1~4 分的语义亲和度量级。
    """
    if edge_bytes <= 0:
        return 0.0
    transfer_cycles = math.ceil(edge_bytes / _EVALUATOR_BANDWIDTH_BYTES_PER_CYCLE)
    return math.log1p(transfer_cycles / _COMMUNICATION_REFERENCE_CYCLES)


def _would_create_partition_cycle(
    features: GraphFeatures,
    left: SemanticBlock,
    right: SemanticBlock,
) -> bool:
    """检测会引入 block DAG 环的非凸合并。

    算法：把候选块视为集合 C，从 C 的所有外部后继开始 DFS。如果 DFS 能走回 C
    中的任意节点，说明存在 C 之外的“环外路径”，合并后会形成环。
    """
    candidate = set(left.nodes) | set(right.nodes)
    seen: set[int] = set()
    stack = [
        successor
        for node in candidate
        for successor in features.succs[node]
        if successor not in candidate
    ]
    while stack:
        node = stack.pop()
        if node in candidate:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(features.succs[node])
    return False


def _singleton_repair_score(
    left: SemanticBlock,
    right: SemanticBlock,
    graph_features: GraphFeatures,
    node_features: dict[int, NodeFeature],
    max_ops: int,
    max_cycles: int,
) -> float:
    """严格 motif 合并后用于清扫 singleton 块的更宽松打分。

    本步对 fan-in/fan-out 比 ``_merge_blocks`` 更宽容（用 log2 而不是硬墙），
    但仍然为并行损失付出代价。它专门清理孤立的“单算子块”，避免严格 motif
    把一些显然后处理的小算子漏在外面。
    """
    if len(left.nodes) + len(right.nodes) > max_ops:
        return float("-inf")
    if left.cycles + right.cycles > max_cycles:
        return float("-inf")

    edge_pairs = _edges_between(graph_features, left, right)
    if not edge_pairs:
        return float("-inf")

    # 用最重的接触边作为语义代表，通信奖励仍按所有跨越张量求和。
    source, target = max(
        edge_pairs,
        key=lambda edge: graph_features.edge_sizes.get(edge, 0),
    )
    tail = node_features[source]
    head = node_features[target]
    if tail.role == "ELEMENTWISE" and head.role == "COMPUTE":
        return float("-inf")
    if tail.pipe != head.pipe and not _explicit_stage_pair(tail, head):
        return float("-inf")
    if _would_create_partition_cycle(graph_features, left, right):
        return float("-inf")

    edge_bytes = _crossing_bytes(graph_features, left, right, edge_pairs)
    communication_saved = _communication_saved(edge_bytes)
    locality_gain = 1.5 if left.output_tensors & right.input_tensors else 0.0
    # 用多边中的最大亲和度作为代表：只要有一条边强相关就值得合并。
    semantic_gain = max(
        _semantic_affinity(node_features[src], node_features[dst])
        for src, dst in edge_pairs
    )
    singleton_gain = 1.0 if len(left.nodes) == 1 or len(right.nodes) == 1 else 0.0
    resource_diversity = 0.8 if tail.pipe != head.pipe else -0.2

    # fan-in/out 仍然重要，但本步不把每个汇合都当成绝对硬墙。大张量复用 +
    # 强语义可以“买通”一定的结构惩罚。log2 让惩罚幅度比硬墙温和。
    fan_loss = math.log2(1 + max(0, tail.outdegree - 1) + max(0, head.indegree - 1))
    boundary_penalty = 0.35 * (tail.boundary_score + head.boundary_score)
    serial_penalty = 0.5 if tail.role == head.role == "COMPUTE" else 0.0
    return (
        communication_saved
        + locality_gain
        + semantic_gain
        + singleton_gain
        + resource_diversity
        - fan_loss
        - boundary_penalty
        - serial_penalty
    )


def repair_singleton_blocks(
    features: GraphFeatures,
    blocks: dict[int, SemanticBlock],
    node_features: dict[int, NodeFeature],
    max_ops: int,
    max_cycles: int,
) -> None:
    """合并严格语义阶段后剩余的有利 singleton 块。

    本修复阶段故意与 ``semantic_partition`` 默认路径分离。当 isolated 单算子
    块过于保守时它是有益的实验；但有可能在 fan-in/fan-out 周围降低并行度。
    最多迭代 2 轮，没有变化就提前退出。
    """
    for _ in range(2):
        owner = _owner_map(blocks)
        changed = False
        for node in features.topo_order:
            block_id = owner.get(node)
            if block_id not in blocks:
                continue
            block = blocks[block_id]
            if len(block.nodes) != 1:
                continue

            # 对当前 singleton 的所有前驱/后继候选打分，取分最高的尝试合并。
            candidates: list[tuple[float, int, int, str]] = []
            for predecessor in sorted(features.preds[node]):
                left_id = owner.get(predecessor)
                if left_id is None or left_id == block_id or left_id not in blocks:
                    continue
                left = blocks[left_id]
                score = _singleton_repair_score(
                    left,
                    block,
                    features,
                    node_features,
                    max_ops,
                    max_cycles,
                )
                candidates.append((score, left_id, block_id, "pred"))
            for successor in sorted(features.succs[node]):
                right_id = owner.get(successor)
                if right_id is None or right_id == block_id or right_id not in blocks:
                    continue
                right = blocks[right_id]
                score = _singleton_repair_score(
                    block,
                    right,
                    features,
                    node_features,
                    max_ops,
                    max_cycles,
                )
                candidates.append((score, block_id, right_id, "succ"))
            if not candidates:
                continue

            score, left_id, right_id, _ = max(candidates, key=lambda item: item[0])
            if score <= 0 or left_id not in blocks or right_id not in blocks:
                continue

            left = blocks[left_id]
            right = blocks[right_id]
            edge_pairs = _edges_between(features, left, right)
            source, target = max(
                edge_pairs,
                key=lambda edge: features.edge_sizes.get(edge, 0),
            )
            _merge(left, right, _motif_type(node_features[source], node_features[target]))
            for merged_node in right.nodes:
                owner[merged_node] = left_id
            del blocks[right_id]
            changed = True

        if not changed:
            break


def _build_stage_blocks(features: GraphFeatures) -> tuple[dict[int, SemanticBlock], dict[int, NodeFeature]]:
    """构造每个节点的初始 singleton block + 节点特征向量。"""
    node_features = extract_features(features)
    blocks = {node: _initial_block(feature) for node, feature in node_features.items()}
    return blocks, node_features


def _merge_blocks(
    features: GraphFeatures,
    blocks: dict[int, SemanticBlock],
    node_features: dict[int, NodeFeature],
    max_ops: int,
    max_cycles: int,
) -> None:
    """不断合并分最高的边，仅在邻域内增量更新。

    使用最大堆（用负分）维护所有合法候选合并。每弹出一个候选：
        1. 校验两端 block 仍然有效、并且仍是各自的“首尾算子”；
        2. 校验仍是凸线性连接（不能吸收分支或汇合）；
        3. 若分数 ≤0 则停止——堆顶非正说明不存在收益合并了；
        4. 执行合并，并只把被影响邻域的新边重新入堆。
    """
    owner = {node: node for node in node_features}
    heap: list[tuple[float, int, int, int]] = []
    serial = 0

    def push(source: int, target: int) -> None:
        nonlocal serial
        left_id, right_id = owner[source], owner[target]
        if left_id == right_id or left_id not in blocks or right_id not in blocks:
            return
        left, right = blocks[left_id], blocks[right_id]
        # 合并必须是凸线性 stage 连接。任何分支或汇合都不能吸收：在算子图为
        # DAG 时，吸收分支/汇合仍可能在 block DAG 上引入环。
        if source != left.nodes[-1] or target != right.nodes[0]:
            return
        if any(owner[pred] != left_id for pred in features.preds[target]):
            return
        if any(owner[successor] != right_id for successor in features.succs[source]):
            return
        score = _merge_score(
            left,
            right,
            node_features,
            features.edge_sizes.get((source, target), 0),
            max_ops,
            max_cycles,
        )
        # serial 是入堆序号，保证同分时按入堆顺序处理，结果稳定。
        heapq.heappush(heap, (-score, serial, left_id, right_id))
        serial += 1

    for source in features.topo_order:
        for target in sorted(features.succs[source]):
            push(source, target)

    while heap:
        neg_score, _, left_id, right_id = heapq.heappop(heap)
        if left_id not in blocks or right_id not in blocks:
            continue
        left, right = blocks[left_id], blocks[right_id]
        # 堆中可能残留旧版邻域，校验两端仍是各自块的首尾。
        if owner[left.nodes[-1]] != left_id or owner[right.nodes[0]] != right_id:
            continue
        score = -neg_score
        if score <= 0:
            break
        tail, head = left.nodes[-1], right.nodes[0]
        if right_id not in {owner[node] for node in features.succs[tail]}:
            continue
        _merge(left, right, _motif_type(node_features[tail], node_features[head]))
        for node in right.nodes:
            owner[node] = left_id
        del blocks[right_id]
        # 只把新合并块的外侧邻边重新入堆；块内边不会再生效。
        tail = left.nodes[-1]
        head = left.nodes[0]
        for successor in features.succs[tail]:
            push(tail, successor)
        for predecessor in features.preds[head]:
            push(predecessor, head)


def build_partition_dag(features: GraphFeatures, blocks: dict[int, SemanticBlock]) -> list[Partition]:
    """把语义块转换成紧凑的 ``Partition`` 对象。"""
    from .algorithm_common import Partition

    # 块按“首个算子的 id”排序，让分区号尽量与算子拓扑序一致，便于人脑对照。
    ordered = sorted(blocks.values(), key=lambda block: (min(block.nodes), block.id))
    remap = {block.id: index for index, block in enumerate(ordered)}
    owner = {node: remap[block.id] for block in ordered for node in block.nodes}
    partitions: list[Partition] = []
    for index, block in enumerate(ordered):
        partitions.append(
            Partition(
                id=index,
                ops=list(block.nodes),
                cycles=block.cycles,
                rank_u=max(features.rank_u[node] for node in block.nodes),
                pipe_cycles=dict(block.pipe_cycles),
            )
        )
    # 再次扫描所有跨分区边，回填 Partition.preds / succs。
    for source in features.topo_order:
        for target in features.succs[source]:
            source_id, target_id = owner[source], owner[target]
            if source_id != target_id:
                partitions[source_id].succs.add(target_id)
                partitions[target_id].preds.add(source_id)
    return partitions


def semantic_partition(
    features: GraphFeatures,
    max_ops: int = 16,
    max_cycles: int = 20000,
    enable_singleton_repair: bool = False,
) -> list[Partition]:
    """用语义 motif 与合并打分对分析后的图进行分区。

    流程：每个算子先成 singleton block → ``_merge_blocks`` 做严格 motif 合并
    → 可选地启用 singleton 修复。``enable_singleton_repair=True`` 时会保留
    修复前的块副本：两块各自合法，合并后却可能在某个非凸区域通过另一条路径
    产生环，这种情况下回退到修复前结果（用 ``_topological_order`` 检测）。
    """
    if max_ops < 1 or max_cycles < 1:
        raise ValueError("partition limits must be positive")
    blocks, node_features = _build_stage_blocks(features)
    _merge_blocks(features, blocks, node_features, max_ops, max_cycles)
    if enable_singleton_repair:
        # 修复阶段比严格合并更宽松。保留检查点：两块各自合法，合并后却可能
        # 在某个非凸区域通过另一条路径产生环，因此需要可回退。
        blocks_before_repair = deepcopy(blocks)
        repair_singleton_blocks(features, blocks, node_features, max_ops, max_cycles)
        repaired = build_partition_dag(features, blocks)
        from .algorithm_common import _topological_order

        try:
            _topological_order(
                [partition.id for partition in repaired],
                {partition.id: set(partition.preds) for partition in repaired},
                {partition.id: set(partition.succs) for partition in repaired},
            )
        except ValueError:
            # 修复引入了环 → 回退到严格合并版本。
            blocks = blocks_before_repair
    return build_partition_dag(features, blocks)


def semantic_partition_with_singleton_repair(
    features: GraphFeatures,
    max_ops: int = 16,
    max_cycles: int = 20000,
) -> list[Partition]:
    """显式开启 singleton 修复的分区入口。"""
    return semantic_partition(
        features,
        max_ops=max_ops,
        max_cycles=max_cycles,
        enable_singleton_repair=True,
    )


__all__ = [
    "NodeFeature",
    "SemanticBlock",
    "extract_features",
    "repair_singleton_blocks",
    "semantic_partition",
    "semantic_partition_with_singleton_repair",
]
