# ============================================================
# 人工智能工具信息 | AI Tool Information
#   工具名称 (Tool Name)        : GLM-5.2
#   版本/型号 (Version/Model)   : GLM-5.2
#   开发机构/公司 (Developer)    : 智谱AI (Zhipu AI / zai-org)
#   版本颁布日期 (Release Date) : 2026-06-13
#   声明：本程序及代码是在人工智能工具辅助下完成的
# ============================================================
"""从原始计算图中提取算子语义特征。

这个文件只负责“看懂图”，不负责生成最终的切图方案或 Core 调度方案。
后续的 partition、buffer 构造和调度算法可以直接读取本文件生成的 JSON。

输出结构（schema_version = "semantic-features-v1"）：
    * graph: 算子/张量/边总数 + 拓扑序 + role/family/pipe 统计 + 候选边统计；
    * operators: 每个算子的 role/family/resource/pipe/cycles/fan_in/fan_out/
      dependency_depth 等可读特征；
    * semantic_edges: 每条 COPY 收缩后的 candidate 依赖边的 fusion_score 与
      可解释 reason 列表。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


COPY_OPS = {"COPY_IN", "COPY_OUT"}
ELEMENTWISE_OPS = {
    "ADD", "SUB", "MUL", "DIV", "RELU", "EXP", "SIGMOID", "NEG", "SQRT",
}


def _role(op_name: str, pipe: str) -> tuple[str, str, str]:
    """根据操作名和流水线给出初步语义标签。

    这里不尝试推断完整的数学算子，只使用题目数据中稳定存在的 op/pipe
    字段。标签足够支持第一版的融合和切分启发式。

    返回 (role, family, resource)：
        * role     —— compute/elementwise/reduction/communication 五大类之一；
        * family   —— 输出更细的算子名（matmul/convolution/...），用于诊断；
        * resource —— MATRIX/VECTOR/DMA 三种之一，与硬件资源族对齐。
    """
    if op_name in COPY_OPS:
        return "communication", "copy", "DMA"
    if op_name == "MATMUL":
        return "compute", "matmul", "MATRIX"
    if op_name == "CONV":
        return "compute", "convolution", "MATRIX"
    if op_name == "REDUCE":
        return "reduction", "reduction", "VECTOR"
    if op_name in ELEMENTWISE_OPS:
        return "elementwise", "elementwise", "VECTOR"
    if pipe == "PIPE_M":
        # 没有 op 名时退化到 pipe 启发：PIPE_M 多为矩阵类。
        return "compute", "generic_compute", "MATRIX"
    if pipe == "PIPE_V":
        return "elementwise", "generic_vector", "VECTOR"
    if pipe in {"PIPE_MTE2", "PIPE_MTE3"}:
        # MTE 是 memory transfer engine，归为通信类。
        return "communication", "memory", "DMA"
    return "compute", "unknown", "UNKNOWN"


def _graph_views(graph: dict[str, Any]) -> tuple[dict[int, dict], dict[int, dict], dict[int, set[int]], dict[int, set[int]]]:
    """构造算子、张量以及算子级依赖关系。

    额外把“op→tensor→op”二跳路径直接连成 op→op 边，这样后续拓扑排序与
    COPY 收缩可以直接基于算子图进行。
    """
    ops = {int(item["id"]): item for item in graph.get("ops", [])}
    tensors = {int(item["id"]): item for item in graph.get("tensors", [])}
    preds = {op_id: set() for op_id in ops}
    succs = {op_id: set() for op_id in ops}
    producers: dict[int, set[int]] = defaultdict(set)
    consumers: dict[int, set[int]] = defaultdict(set)

    for edge in graph.get("edges", []):
        source, target = int(edge["source"]), int(edge["target"])
        if source in ops and target in ops:
            succs[source].add(target)
            preds[target].add(source)
        elif source in ops and target in tensors:
            producers[target].add(source)
        elif source in tensors and target in ops:
            consumers[source].add(target)

    # 把 tensor 中转依赖桥接成直接的 op→op 边。
    for tensor_id, source_ids in producers.items():
        for source in source_ids:
            for target in consumers.get(tensor_id, ()):
                if source != target:
                    succs[source].add(target)
                    preds[target].add(source)
    return ops, tensors, preds, succs


def _topological_depth(op_ids: list[int], preds: dict[int, set[int]], succs: dict[int, set[int]]) -> tuple[list[int], dict[int, int]]:
    """计算拓扑序和从输入开始的依赖深度。

    依赖深度即节点在 DAG 中的最长前驱链长度：root = 0，其余 = max(前驱) + 1。
    """
    indegree = {op_id: len(preds[op_id]) for op_id in op_ids}
    depth = {op_id: 0 for op_id in op_ids}
    # 就绪队列用 deque + sorted，保证同一层按 op_id 升序处理，结果稳定。
    ready = deque(sorted(op_id for op_id in op_ids if indegree[op_id] == 0))
    order: list[int] = []
    while ready:
        current = ready.popleft()
        order.append(current)
        for target in sorted(succs[current]):
            depth[target] = max(depth[target], depth[current] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    # 原始题目保证 DAG；保留一个明确的失败信号，避免静默产生错误特征。
    if len(order) != len(op_ids):
        raise ValueError("算子依赖图存在环，无法计算语义深度")
    return order, depth


def _contract_copy_edges(ops: dict[int, dict], succs: dict[int, set[int]]) -> dict[int, set[int]]:
    """跳过 COPY 节点，生成非 COPY 算子之间的候选融合边。

    对每个非 COPY 算子做 BFS：沿 succs 向下走，跳过 COPY 后落到下一个非
    COPY 节点，记录为一条 candidate 融合边。这等价于评测器把 COPY 看作
    边界后用户方案的“跨分区依赖”集合。
    """
    eligible = {op_id for op_id, op in ops.items() if op.get("op") not in COPY_OPS}
    contracted = {op_id: set() for op_id in eligible}
    for source in sorted(eligible):
        queue = list(sorted(succs[source], reverse=True))
        visited: set[int] = set()
        while queue:
            target = queue.pop()
            if target in eligible:
                if target != source:
                    contracted[source].add(target)
                continue
            if target not in visited:
                visited.add(target)
                queue.extend(sorted(succs.get(target, ()), reverse=True))
    return contracted


def _fusion_score(source: dict, target: dict) -> tuple[float, list[str]]:
    """给一条候选依赖边打简单、可解释的融合分。

    打分规则（每条记录一条 reason）：
        * elementwise→elementwise                 +4.0；连续逐元素；
        * compute→elementwise/reduction           +4.0；计算后轻量后处理；
        * reduction→elementwise                   +3.0；归约后逐元素；
        * 同角色                                   +2.0；
        * 同 resource（非 UNKNOWN）                +1.0；可避免资源切换；
        * source fan_out > 1                      -2.0；保留分支；
        * target fan_in > 1                       -1.0；避免汇聚串行；
        * matmul→matmul                           -1.0；连续矩阵更适合并行。
    最终阈值 ≥4.0 → fuse，否则 cut。
    """
    source_role, source_family = source["role"], source["family"]
    target_role, target_family = target["role"], target["family"]
    score = 0.0
    reasons: list[str] = []
    if source_role == "elementwise" and target_role == "elementwise":
        score += 4.0
        reasons.append("连续逐元素算子")
    elif source_role == "compute" and target_role in {"elementwise", "reduction"}:
        score += 4.0
        reasons.append("计算结果紧接轻量后处理")
    elif source_role == "reduction" and target_role == "elementwise":
        score += 3.0
        reasons.append("归约结果紧接逐元素后处理")
    elif source_role == target_role:
        score += 2.0
        reasons.append("算子角色一致")

    if source["resource"] == target["resource"] and source["resource"] != "UNKNOWN":
        score += 1.0
        reasons.append("执行资源类型一致")
    if source["fan_out"] > 1:
        score -= 2.0
        reasons.append("源算子存在分支，保留并行性")
    if target["fan_in"] > 1:
        score -= 1.0
        reasons.append("目标算子存在多路汇聚")
    if source_family == target_family == "matmul":
        score -= 1.0
        reasons.append("连续矩阵计算可能更适合并行")
    decision = "fuse" if score >= 4.0 else "cut"
    return score, reasons + [f"建议:{decision}"]


def analyze_graph(graph: dict[str, Any], source: str | None = None) -> dict[str, Any]:
    """提取一张计算图的语义特征和可解释的融合边评分。

    返回结构详见模块 docstring。``source`` 仅写到 ``source`` 字段，方便后续
    JSON 报告溯源。
    """
    ops, tensors, preds, succs = _graph_views(graph)
    order, depth = _topological_depth(sorted(ops), preds, succs)
    # 同时整理 tensor→op 关系，方便下面给每个 op 算 input_tensors /
    # output_tensors 与字节数。
    tensor_inputs: dict[int, list[int]] = defaultdict(list)
    tensor_outputs: dict[int, list[int]] = defaultdict(list)
    for edge in graph.get("edges", []):
        source_id, target_id = int(edge["source"]), int(edge["target"])
        if source_id in tensors and target_id in ops:
            tensor_inputs[target_id].append(source_id)
        elif source_id in ops and target_id in tensors:
            tensor_outputs[source_id].append(target_id)

    features: dict[int, dict[str, Any]] = {}
    for op_id in order:
        op = ops[op_id]
        role, family, resource = _role(op.get("op", ""), op.get("pipe", ""))
        input_ids = sorted(set(tensor_inputs[op_id]))
        output_ids = sorted(set(tensor_outputs[op_id]))
        input_bytes = sum(int(tensors[tensor_id].get("size", 0)) for tensor_id in input_ids)
        output_bytes = sum(int(tensors[tensor_id].get("size", 0)) for tensor_id in output_ids)
        fan_in = len(preds[op_id])
        fan_out = len(succs[op_id])
        features[op_id] = {
            "id": op_id,
            "op": op.get("op"),
            "role": role,
            "family": family,
            "resource": resource,
            "pipe": op.get("pipe"),
            "cycles": int(op.get("cycles", 0)),
            "input_tensors": [
                {"id": tensor_id, "pos": tensors[tensor_id].get("pos"), "size": int(tensors[tensor_id].get("size", 0))}
                for tensor_id in input_ids
            ],
            "output_tensors": [
                {"id": tensor_id, "pos": tensors[tensor_id].get("pos"), "size": int(tensors[tensor_id].get("size", 0))}
                for tensor_id in output_ids
            ],
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "fan_in": fan_in,
            "fan_out": fan_out,
            "dependency_depth": depth[op_id],
            "is_branch": fan_out > 1,
            "is_join": fan_in > 1,
            # parallelism_hint 给上层可视化用：直观显示该算子处于 branch/join
            # 还是 chain 形态。
            "parallelism_hint": "branch" if fan_out > 1 else "join" if fan_in > 1 else "chain",
        }

    contracted = _contract_copy_edges(ops, succs)
    semantic_edges: list[dict[str, Any]] = []
    # 对每条 COPY 收缩后的候选边打分，把评分、理由、decision 与通信字节都记下。
    for source_id in sorted(contracted):
        for target_id in sorted(contracted[source_id]):
            score, reasons = _fusion_score(features[source_id], features[target_id])
            semantic_edges.append({
                "source": source_id,
                "target": target_id,
                "score": score,
                "decision": "fuse" if score >= 4.0 else "cut",
                "reasons": reasons,
                "communication_bytes": features[source_id]["output_bytes"],
            })

    role_counts = Counter(item["role"] for item in features.values())
    family_counts = Counter(item["family"] for item in features.values())
    pipe_counts = Counter(item["pipe"] for item in features.values())
    return {
        "schema_version": "semantic-features-v1",
        "source": source,
        "graph": {
            "operator_count": len(ops),
            "tensor_count": len(tensors),
            "edge_count": len(graph.get("edges", [])),
            "non_copy_operator_count": sum(item.get("op") not in COPY_OPS for item in ops.values()),
            "topological_order": order,
            "role_counts": dict(role_counts),
            "family_counts": dict(family_counts),
            "pipe_counts": dict(pipe_counts),
            "candidate_fusion_edges": sum(edge["decision"] == "fuse" for edge in semantic_edges),
            "candidate_cut_edges": sum(edge["decision"] == "cut" for edge in semantic_edges),
        },
        "operators": [features[op_id] for op_id in order],
        "semantic_edges": semantic_edges,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：读取图 → 输出语义特征 JSON。"""
    parser = argparse.ArgumentParser(description="提取计算图中的算子语义特征")
    parser.add_argument("graph", type=Path, help="输入计算图 JSON")
    parser.add_argument("-o", "--output", type=Path, help="输出语义特征 JSON")
    args = parser.parse_args(argv)
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    result = analyze_graph(graph, str(args.graph))
    output = args.output or args.graph.with_name(f"{args.graph.stem}_semantic_features.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"OK: operators={len(result['operators'])}, semantic_edges={len(result['semantic_edges'])}, output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
