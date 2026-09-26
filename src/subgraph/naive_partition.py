# ============================================================
# 人工智能工具信息 | AI Tool Information
#   工具名称 (Tool Name)        : GLM-5.2
#   版本/型号 (Version/Model)   : GLM-5.2
#   开发机构/公司 (Developer)    : 智谱AI (Zhipu AI / zai-org)
#   版本颁布日期 (Release Date) : 2026-06-13
#   声明：本程序及代码是在人工智能工具辅助下完成的
# ============================================================
"""原始的保守贪心分区器。

本模块保留语义分区器上线之前的实现，作为可选的对照策略。它只在边是
直链连接（即单一前驱 + 单一后继）并且分区上限得到尊重时，把 ELEMENTWISE
算子并入其前驱所在分区。

之所以**只对 ELEMENTWISE 做这种合并**：ELEMENTWISE 通常只是把前驱的计算
结果做一次轻量后处理（例如激活、加常数），合并后能省掉一次子图边界与跨
核通信；而 MATMUL/REDUCE 等重计算算子之间则需要保留并行度，合并反而会
把可并行的算子串行化，因此不在贪心路径中合并。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .algorithm_common import GraphFeatures, Partition


def naive_partition(
    features: GraphFeatures,
    max_ops: int = 16,
    max_cycles: int = 20000,
) -> list[Partition]:
    """按历史的单遍贪心规则构建分区。

    算法步骤：
        1. 按拓扑序遍历每个算子；
        2. 若当前算子是 ELEMENTWISE 且前驱唯一、且前驱也只有一个后继
           （即两者处于同一条直链上），则尝试并入前驱所在分区；
        3. 并入前需校验 ``max_ops`` / ``max_cycles`` 是否仍满足；
        4. 否则新建只含当前算子的分区。
    最后再扫描一遍所有跨分区边，建立分区 DAG 的 ``preds`` / ``succs``。
    """
    if max_ops < 1 or max_cycles < 1:
        raise ValueError("partition limits must be positive")

    # 延迟导入以避免在只是引入该“可选实现”的模块时与 q2_algorithm 形成循环依赖。
    from .algorithm_common import Partition

    partitions: list[Partition] = []
    op_to_partition: dict[int, int] = {}
    for node in features.topo_order:
        op = features.op_by_id[node]
        pred_ids = features.preds[node]
        candidate = None
        # 直链 ELEMENTWISE 才尝试并入前驱；这是历史贪心规则的核心保守条件。
        if len(pred_ids) == 1 and features.semantic[node] == "ELEMENTWISE":
            pred = next(iter(pred_ids))
            pred_partition = op_to_partition.get(pred)
            # 第二个守卫：前驱也只能有这一个后继，否则融合会破坏并行性。
            if pred_partition is not None and len(features.succs[pred]) == 1:
                partition = partitions[pred_partition]
                cycles = int(op.get("cycles", 0))
                if (
                    len(partition.ops) < max_ops
                    and partition.cycles + cycles <= max_cycles
                ):
                    candidate = partition
        # 不满足融合条件则单独成块，保证每个算子都被分配。
        if candidate is None:
            candidate = Partition(
                id=len(partitions),
                ops=[],
                cycles=0,
                rank_u=features.rank_u[node],
            )
            partitions.append(candidate)

        cycles = int(op.get("cycles", 0))
        candidate.ops.append(node)
        candidate.cycles += cycles
        pipe = str(op.get("pipe", "UNKNOWN"))
        candidate.pipe_cycles[pipe] = candidate.pipe_cycles.get(pipe, 0) + cycles
        # 关键路径估计取分区内所有算子的最大 rank_u；调度器据此排序。
        candidate.rank_u = max(candidate.rank_u, features.rank_u[node])
        op_to_partition[node] = candidate.id

    # 第二遍：根据 op 级后继关系构造分区 DAG 边。
    for source in features.topo_order:
        for target in features.succs[source]:
            source_partition = op_to_partition[source]
            target_partition = op_to_partition[target]
            if source_partition != target_partition:
                partitions[source_partition].succs.add(target_partition)
                partitions[target_partition].preds.add(source_partition)
    return partitions


__all__ = ["naive_partition"]
