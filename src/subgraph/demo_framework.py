"""Framework entry point for the graph-planning demo.

Graph analysis and the concrete scheduling strategies live in
``demo_algorithm``.  This module owns the public pipeline and command-line
boundary only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .demo_algorithm import (
    GraphFeatures,
    analyze_graph,
    classify_op,
    describe_graph_pattern,
)
from .interfaces import AlgorithmResult


def build_plan(
    graph: dict[str, Any],
    num_cores: int = 4,
    scenario: str = "q2",
    features: GraphFeatures | None = None,
) -> AlgorithmResult:
    """Build a plan using the selected concrete graph algorithm."""
    if features is None:
        features = analyze_graph(graph)
    from .demo_algorithm import build_algorithm_plan

    return build_algorithm_plan(features, num_cores, scenario)


def load_graph(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        graph = json.load(stream)
    if not isinstance(graph, dict):
        raise ValueError("graph JSON must contain an object")
    return graph


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="极简统一计算图切分与多核调度框架")
    parser.add_argument("graph", type=Path)
    parser.add_argument("-n", "--num-cores", type=int, default=4)
    parser.add_argument("--scenario", choices=("q1", "q2", "q3"), default="q2")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args(argv)
    graph = load_graph(args.graph)
    result = build_plan(graph, args.num_cores, args.scenario)
    output = args.output or args.graph.with_name(f"{args.graph.stem}_multicore_res.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result.plan, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        f"OK: scenario={args.scenario}, ops={len(result.plan['node_to_subgraph'])}, "
        f"subgraphs={len(set(result.plan['node_to_subgraph'].values()))}, output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
