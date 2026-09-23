"""Baseline 1：拓扑排序 + 均匀切块 + 一核一块。

该基线对应 ``docs/demo_solution.md`` 中的 Baseline 1：

1. 忽略原图中的 COPY_IN/COPY_OUT，只对核内计算操作排序；
2. 将计算操作按确定性的拓扑序排列；
3. 尽量均匀地切成 ``K`` 个连续子图；
4. 第 i 个非空子图分配给第 i 个核心，并作为该核心唯一的 Task。

这不是通信感知或负载感知算法，作用是提供一个简单、可复现、格式合法的
性能基线。评估器会根据输出方案自动补充边界 COPY 和核内调度。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_stub_helpers():
    """加载题目提供的图依赖构造和方案校验逻辑。"""
    # 当前目录布局：
    #   artifacts/code/stub_multicore_cut_and_schedule.py
    #   src/subgraph/baseline1.py
    repo_root = Path(__file__).resolve().parents[2]
    artifacts_code = repo_root / "artifacts" / "code"
    if not artifacts_code.is_dir():
        raise FileNotFoundError(
            f"artifacts/code not found: {artifacts_code}; "
            "please check the project layout"
        )
    if str(artifacts_code) not in sys.path:
        sys.path.insert(0, str(artifacts_code))

    from stub_multicore_cut_and_schedule import (  # type: ignore
        EXCLUDED_COPY_TYPES,
        _build_op_adjacency,
        _contract_excluded_copy_nodes,
        validate_multicore_plan,
    )

    return (
        EXCLUDED_COPY_TYPES,
        _build_op_adjacency,
        _contract_excluded_copy_nodes,
        validate_multicore_plan,
    )


def _topological_order(graph: dict[str, Any]) -> list[int]:
    """对非 COPY 操作生成确定性的拓扑序。

    原图依赖可能经过 COPY_IN/COPY_OUT；先使用 stub 的收缩逻辑跳过这些节点，
    保留非 COPY 操作之间的可达依赖。
    """
    (
        excluded_copy_types,
        build_op_adjacency,
        contract_excluded_copy_nodes,
        _,
    ) = _load_stub_helpers()

    op_by_id = {op["id"]: op for op in graph.get("ops", [])}
    eligible = sorted(
        op_id
        for op_id, op in op_by_id.items()
        if op.get("op") not in excluded_copy_types
    )
    _, full_succs = build_op_adjacency(graph)
    _, succs = contract_excluded_copy_nodes(eligible, full_succs)

    indegree = {op_id: 0 for op_id in eligible}
    for source in eligible:
        for target in succs[source]:
            indegree[target] += 1

    ready = [op_id for op_id in eligible if indegree[op_id] == 0]
    ready.sort()
    order: list[int] = []

    while ready:
        op_id = ready.pop(0)
        order.append(op_id)
        for successor in sorted(succs[op_id]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()

    if len(order) != len(eligible):
        unresolved = sorted(set(eligible) - set(order))
        raise ValueError(
            "non-COPY operation graph contains a cycle; "
            f"unresolved operation ids={unresolved[:20]}"
        )
    return order


def _balanced_chunks(items: list[int], num_chunks: int) -> list[list[int]]:
    """将列表均匀切分，块大小最多相差 1。"""
    if num_chunks < 1:
        raise ValueError("num_chunks must be at least 1")

    chunk_count = min(num_chunks, len(items)) if items else 0
    if chunk_count == 0:
        return []

    base_size, remainder = divmod(len(items), chunk_count)
    chunks: list[list[int]] = []
    start = 0
    for index in range(chunk_count):
        size = base_size + (1 if index < remainder else 0)
        chunks.append(items[start:start + size])
        start += size
    return chunks


def build_baseline1_plan(
    graph: dict[str, Any], num_cores: int
) -> dict[str, Any]:
    """生成 Baseline 1 的标准多核方案。"""
    if num_cores < 1:
        raise ValueError("num_cores must be at least 1")

    topo_order = _topological_order(graph)
    chunks = _balanced_chunks(topo_order, num_cores)

    node_to_subgraph: dict[str, int] = {}
    core_schedules: list[list[int]] = [[] for _ in range(num_cores)]

    for subgraph_id, chunk in enumerate(chunks):
        for op_id in chunk:
            node_to_subgraph[str(op_id)] = subgraph_id
        core_schedules[subgraph_id].append(subgraph_id)

    return {
        "node_to_subgraph": node_to_subgraph,
        "core_schedules": core_schedules,
    }


def load_graph(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        graph = json.load(stream)
    if not isinstance(graph, dict):
        raise ValueError("graph JSON must contain an object")
    return graph


def write_plan(path: Path, plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Baseline 1: 拓扑排序、均匀切块、一核一块"
    )
    parser.add_argument("graph", type=Path, help="输入计算图 JSON")
    parser.add_argument(
        "-n",
        "--num-cores",
        type=int,
        default=4,
        help="核心数量，默认 4",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="输出方案 JSON；默认写到 <graph>_multicore_res.json",
    )
    args = parser.parse_args(argv)

    graph = load_graph(args.graph)
    plan = build_baseline1_plan(graph, args.num_cores)

    *_, validate_multicore_plan = _load_stub_helpers()
    validate_multicore_plan(graph, plan)

    output = args.output or args.graph.with_name(
        f"{args.graph.stem}_multicore_res.json"
    )
    write_plan(output, plan)
    print(
        "OK: generated baseline1 plan; "
        f"ops={len(plan['node_to_subgraph'])}, "
        f"subgraphs={len(set(plan['node_to_subgraph'].values()))}, "
        f"cores={args.num_cores}, output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
