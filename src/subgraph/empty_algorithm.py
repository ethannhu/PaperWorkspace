"""多核切图算法空模板。

这个脚本只负责生成一份格式合法的最小方案：

* 所有非 COPY 操作放入同一个子图 0；
* 子图 0 分配到核心 0；
* 其余核心暂时为空。

后续实现算法时，主要替换 ``build_empty_plan``，保留方案格式和
``validate_multicore_plan`` 校验即可。评估器会自动补充边界 COPY、执行核内
调度，并计算最终指标。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_stub_helpers():
    """加载题目提供的方案校验逻辑，不修改 artifacts 中的文件。"""
    # artifacts 与本脚本位于同一个 src/subgraph 目录下：
    # src/subgraph/empty_algorithm.py
    # src/subgraph/artifacts/code/stub_multicore_cut_and_schedule.py
    artifacts_code = Path(__file__).resolve().parent / "artifacts" / "code"
    if str(artifacts_code) not in sys.path:
        sys.path.insert(0, str(artifacts_code))

    from stub_multicore_cut_and_schedule import (  # type: ignore
        EXCLUDED_COPY_TYPES,
        validate_multicore_plan,
    )

    return EXCLUDED_COPY_TYPES, validate_multicore_plan


def build_empty_plan(graph: dict[str, Any], num_cores: int) -> dict[str, Any]:
    """生成空算法的基线方案。

    该方案不做任何切图优化，只把所有可调度计算操作合并成一个子图。
    ``num_cores`` 仍然保留在输出中，便于直接提交给不同核数的评估器。
    """
    if num_cores < 1:
        raise ValueError("num_cores must be at least 1")

    excluded_copy_types, _ = _load_stub_helpers()
    eligible_ops = sorted(
        op["id"]
        for op in graph.get("ops", [])
        if op.get("op") not in excluded_copy_types
    )

    # JSON 对象的键最终会被写成字符串；评估器同时接受数字字符串。
    node_to_subgraph = {str(op_id): 0 for op_id in eligible_ops}
    core_schedules = [[0]] + [[] for _ in range(num_cores - 1)]

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
    parser = argparse.ArgumentParser(description="生成多核切图算法空模板方案")
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
    plan = build_empty_plan(graph, args.num_cores)

    # 复用 stub 的完整合法性检查，确保后续替换算法时尽早发现格式问题。
    _, validate_multicore_plan = _load_stub_helpers()
    validate_multicore_plan(graph, plan)

    output = args.output or args.graph.with_name(
        f"{args.graph.stem}_multicore_res.json"
    )
    write_plan(output, plan)
    print(
        "OK: generated empty plan; "
        f"ops={len(plan['node_to_subgraph'])}, "
        f"subgraphs=1, cores={args.num_cores}, output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
