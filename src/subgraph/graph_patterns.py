"""Recognise the seven computation-graph layouts described in ``docs/``.

The classifier intentionally uses graph properties rather than case ids.  It
therefore also works for an unseen graph that follows one of the documented
layouts.  Rules are ordered from the most distinctive motifs to the broad
mixed fallback and each result includes the measurements that led to it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .demo_framework import GraphFeatures, analyze_graph


class GraphPattern(StrEnum):
    """The seven fine-grained graph classes from ``case_graph_patterns.md``."""

    MIXED_MLP_REDUCE = "mixed_mlp_reduce"
    SHALLOW_WIDE_COMPUTE_ACTIVATION = "shallow_wide_matmul_relu"
    CNN_RESIDUAL = "cnn_residual"
    GATED_SIGMOID_MLP = "gated_sigmoid_mlp"
    ATTENTION_NORMALIZE = "attention_normalize"
    NARROW_DEEP_REDUCE_RELU_ADD = "narrow_deep_reduce_relu_add"
    WIDE_MATMUL_ADD = "wide_matmul_add"


class GraphPatternFamily(StrEnum):
    """Coarse routing classes used by the algorithm framework."""

    WIDE = "wide"
    NARROW = "narrow"
    MIXED = "mixed"
    COMPLEX = "complex"


_DISPLAY_NAMES = {
    GraphPattern.MIXED_MLP_REDUCE: "小/中型混合 MLP-Reduce 图",
    GraphPattern.SHALLOW_WIDE_COMPUTE_ACTIVATION: "浅层宽并行 MatMul-ReLU 图",
    GraphPattern.CNN_RESIDUAL: "CNN / Residual 卷积块",
    GraphPattern.GATED_SIGMOID_MLP: "Gated / Sigmoid-MLP 块",
    GraphPattern.ATTENTION_NORMALIZE: "Attention / Normalize 风格复杂链",
    GraphPattern.NARROW_DEEP_REDUCE_RELU_ADD: "极窄超深 Reduce-Relu-Add 链",
    GraphPattern.WIDE_MATMUL_ADD: "极宽 MatMul-Add 批处理图",
}


_FAMILY_BY_PATTERN = {
    GraphPattern.SHALLOW_WIDE_COMPUTE_ACTIVATION: GraphPatternFamily.WIDE,
    GraphPattern.GATED_SIGMOID_MLP: GraphPatternFamily.WIDE,
    GraphPattern.WIDE_MATMUL_ADD: GraphPatternFamily.WIDE,
    GraphPattern.NARROW_DEEP_REDUCE_RELU_ADD: GraphPatternFamily.NARROW,
    GraphPattern.MIXED_MLP_REDUCE: GraphPatternFamily.MIXED,
    GraphPattern.CNN_RESIDUAL: GraphPatternFamily.COMPLEX,
    GraphPattern.ATTENTION_NORMALIZE: GraphPatternFamily.COMPLEX,
}


@dataclass(frozen=True)
class GraphPatternReport:
    """Classification plus the feature values needed to audit the decision."""

    pattern: GraphPattern
    display_name: str
    family: GraphPatternFamily
    reason: str
    operator_count: int
    depth: int
    max_width: int
    branch_ratio: float
    join_ratio: float
    operator_ratios: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["pattern"] = self.pattern.value
        result["family"] = self.family.value
        return result


_TRACKED_OPS = (
    "MATMUL", "CONV", "ADD", "RELU", "REDUCE", "SUB", "DIV", "MUL", "EXP", "SIGMOID",
)


def _measure(features: GraphFeatures) -> tuple[int, int, int, float, float, dict[str, float]]:
    nodes = features.topo_order
    count = len(nodes)
    if not count:
        return 0, 0, 0, 0.0, 0.0, {op: 0.0 for op in _TRACKED_OPS}
    levels = Counter(features.depth[node] for node in nodes)
    op_counts = Counter(str(features.op_by_id[node].get("op", "")) for node in nodes)
    return (
        count,
        max(levels) + 1,
        max(levels.values()),
        sum(len(features.succs[node]) > 1 for node in nodes) / count,
        sum(len(features.preds[node]) > 1 for node in nodes) / count,
        {op: op_counts[op] / count for op in _TRACKED_OPS},
    )


def classify_features(features: GraphFeatures) -> GraphPatternReport:
    """Classify an already analysed, COPY-contracted operation DAG.

    Thresholds were calibrated against all ``artifacts/data/case_*.json``
    examples.  They are deliberately expressed as ratios and topology measures
    so that graph replication changes neither the class nor the decision.
    """
    count, depth, width, branch, join, ratio = _measure(features)
    mm, conv, add, relu, reduce = (ratio[key] for key in ("MATMUL", "CONV", "ADD", "RELU", "REDUCE"))
    sub, div, mul, exp, sigmoid = (ratio[key] for key in ("SUB", "DIV", "MUL", "EXP", "SIGMOID"))

    if width <= 16 and depth >= 100 and reduce >= 0.15 and relu >= 0.30 and add >= 0.30:
        pattern = GraphPattern.NARROW_DEEP_REDUCE_RELU_ADD
        reason = "层宽极窄且深度很大，Reduce/RELU/ADD 为主"
    elif mm >= 0.25 and mul >= 0.15 and sigmoid >= 0.08:
        pattern = GraphPattern.GATED_SIGMOID_MLP
        reason = "MATMUL、SIGMOID 与 MUL 同时高占比，符合门控 motif"
    elif mm >= 0.30 and add >= 0.30 and mm + add >= 0.68 and relu < 0.10:
        pattern = GraphPattern.WIDE_MATMUL_ADD
        reason = "MATMUL 与 ADD 绝对主导，RELU 很少"
    elif sub + div + exp >= 0.16 and reduce >= 0.10 and branch >= 0.15 and join >= 0.30:
        pattern = GraphPattern.ATTENTION_NORMALIZE
        reason = "SUB/EXP/DIV/REDUCE 归一化链且分支、汇合密集"
    # Some generated wide blocks use CONV in place of MATMUL.  Topology is
    # decisive for this class, while the second clause admits the documented
    # deeper MATMUL-RELU variant without confusing it with residual blocks.
    elif depth <= 8 or (depth <= 32 and mm >= 0.20 and relu >= 0.20):
        pattern = GraphPattern.SHALLOW_WIDE_COMPUTE_ACTIVATION
        reason = "计算图很浅，存在大层宽的可并行 compute-activation 块"
    elif conv + relu + add >= 0.95 and relu >= 0.30 and add >= 0.15:
        pattern = GraphPattern.CNN_RESIDUAL
        reason = "CONV/RELU/ADD（或其残差等价形）几乎覆盖全图"
    else:
        pattern = GraphPattern.MIXED_MLP_REDUCE
        reason = "未命中专属 motif；为中等深度的混合 MLP/Reduce 排布"

    return GraphPatternReport(
        pattern=pattern,
        display_name=_DISPLAY_NAMES[pattern],
        family=_FAMILY_BY_PATTERN[pattern],
        reason=reason,
        operator_count=count,
        depth=depth,
        max_width=width,
        branch_ratio=branch,
        join_ratio=join,
        operator_ratios=ratio,
    )


def classify_graph(graph: dict[str, Any]) -> GraphPatternReport:
    """Analyse a raw input graph and return its seven-class pattern report."""
    return classify_features(analyze_graph(graph))


def family_for_pattern(pattern: GraphPattern) -> GraphPatternFamily:
    """Return the coarse routing family for a fine-grained graph pattern."""
    return _FAMILY_BY_PATTERN[pattern]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="识别计算图的七类算子排布模式")
    parser.add_argument("graph", type=Path, help="输入计算图 JSON")
    parser.add_argument("-o", "--output", type=Path, help="可选的 JSON 报告路径")
    args = parser.parse_args(argv)
    report = classify_graph(json.loads(args.graph.read_text(encoding="utf-8"))).as_dict()
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
