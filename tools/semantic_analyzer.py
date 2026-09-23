"""从原始计算图中提取算子语义特征。

这个文件只负责“看懂图”，不负责生成最终的切图方案或 Core 调度方案。
后续的 partition、buffer 构造和调度算法可以直接读取本文件生成的 JSON。
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
        return "compute", "generic_compute", "MATRIX"
    if pipe == "PIPE_V":
        return "elementwise", "generic_vector", "VECTOR"
    if pipe in {"PIPE_MTE2", "PIPE_MTE3"}:
        return "communication", "memory", "DMA"
    return "compute", "unknown", "UNKNOWN"


def _graph_views(graph: dict[str, Any]) -> tuple[dict[int, dict], dict[int, dict], dict[int, set[int]], dict[int, set[int]]]:
    """构造算子、张量以及算子级依赖关系。"""
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

    for tensor_id, source_ids in producers.items():
        for source in source_ids:
            for target in consumers.get(tensor_id, ()):
                if source != target:
                    succs[source].add(target)
                    preds[target].add(source)
    return ops, tensors, preds, succs


def _topological_depth(op_ids: list[int], preds: dict[int, set[int]], succs: dict[int, set[int]]) -> tuple[list[int], dict[int, int]]:
    """计算拓扑序和从输入开始的依赖深度。"""
    indegree = {op_id: len(preds[op_id]) for op_id in op_ids}
    depth = {op_id: 0 for op_id in op_ids}
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
    """跳过 COPY 节点，生成非 COPY 算子之间的候选融合边。"""
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
    """给一条候选依赖边打简单、可解释的融合分。"""
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
    """提取一张计算图的语义特征和可解释的融合边评分。"""
    ops, tensors, preds, succs = _graph_views(graph)
    order, depth = _topological_depth(sorted(ops), preds, succs)
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
            "parallelism_hint": "branch" if fan_out > 1 else "join" if fan_in > 1 else "chain",
        }

    contracted = _contract_copy_edges(ops, succs)
    semantic_edges: list[dict[str, Any]] = []
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
