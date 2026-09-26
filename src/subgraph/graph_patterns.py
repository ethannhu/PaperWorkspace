# ============================================================
# 人工智能工具信息 | AI Tool Information
#   工具名称 (Tool Name)        : GLM-5.2
#   版本/型号 (Version/Model)   : GLM-5.2
#   开发机构/公司 (Developer)    : 智谱AI (Zhipu AI / zai-org)
#   版本颁布日期 (Release Date) : 2026-06-13
#   声明：本程序及代码是在人工智能工具辅助下完成的
# ============================================================
"""识别 ``docs/`` 中描述的七类计算图布局。

分类器刻意使用图本身的结构与算子比例来判定，而不是 case id。这样对未见
过、但符合同一种 motif 的新图也能正确路由。规则按“最有区分度的 motif 优先
排序、混合形态作为兜底”，每个结果都附带促成判定的度量值，便于事后审计。

七类细粒度模式（``GraphPattern``）会被映射到四类粗粒度路由族
（``GraphPatternFamily`` = WIDE / NARROW / MIXED / COMPLEX），算法框架按
路由族选择对应的分区与调度策略。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .algorithm_common import GraphFeatures, analyze_graph


class GraphPattern(StrEnum):
    """``case_graph_patterns.md`` 中定义的七种细粒度图模式。"""

    MIXED_MLP_REDUCE = "mixed_mlp_reduce"
    SHALLOW_WIDE_COMPUTE_ACTIVATION = "shallow_wide_matmul_relu"
    CNN_RESIDUAL = "cnn_residual"
    GATED_SIGMOID_MLP = "gated_sigmoid_mlp"
    ATTENTION_NORMALIZE = "attention_normalize"
    NARROW_DEEP_REDUCE_RELU_ADD = "narrow_deep_reduce_relu_add"
    WIDE_MATMUL_ADD = "wide_matmul_add"


class GraphPatternFamily(StrEnum):
    """算法框架使用的粗粒度路由族。"""

    WIDE = "wide"
    NARROW = "narrow"
    MIXED = "mixed"
    COMPLEX = "complex"


# 七类模式对应的中文展示名，用于诊断 JSON 与 CLI 输出。
_DISPLAY_NAMES = {
    GraphPattern.MIXED_MLP_REDUCE: "小/中型混合 MLP-Reduce 图",
    GraphPattern.SHALLOW_WIDE_COMPUTE_ACTIVATION: "浅层宽并行 MatMul-ReLU 图",
    GraphPattern.CNN_RESIDUAL: "CNN / Residual 卷积块",
    GraphPattern.GATED_SIGMOID_MLP: "Gated / Sigmoid-MLP 块",
    GraphPattern.ATTENTION_NORMALIZE: "Attention / Normalize 风格复杂链",
    GraphPattern.NARROW_DEEP_REDUCE_RELU_ADD: "极窄超深 Reduce-Relu-Add 链",
    GraphPattern.WIDE_MATMUL_ADD: "极宽 MatMul-Add 批处理图",
}


# 细粒度模式 → 粗粒度路由族映射。算法 main 入口处按家族分派策略树。
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
    """分类结果 + 审计所需的关键度量值。"""

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
        """转 JSON 友好字典：把枚举转成字符串值，便于落盘与人工查看。"""
        result = asdict(self)
        result["pattern"] = self.pattern.value
        result["family"] = self.family.value
        return result


# 七条分类规则中需要统计的算子类型集合。
_TRACKED_OPS = (
    "MATMUL", "CONV", "ADD", "RELU", "REDUCE", "SUB", "DIV", "MUL", "EXP", "SIGMOID",
)


def _measure(features: GraphFeatures) -> tuple[int, int, int, float, float, dict[str, float]]:
    """把 ``GraphFeatures`` 压缩成分类规则使用的度量值。

    返回值含义：
        * 算子总数 count；
        * 最大深度 depth（层数 = 最大 depth + 1）；
        * 最大层宽 width（同一 depth 上算子数的最大值）；
        * 分支比例 branch（多后继算子占比）；
        * 汇合比例 join（多前驱算子占比）；
        * 每种被追踪算子类型的比例字典 ratio。
    """
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
    """对一个已分析、已 COPY 收缩的算子 DAG 写出七类分类报告。

    阈值是针对 ``artifacts/data/case_*.json`` 中所有样例标定得到的。阈值故意
    都用比例和拓扑度量来表示，这样把算子复制多份（例如把 8 个分支变 16 个）
    既不会改动分类，也不会扰动路由决策。
    """
    count, depth, width, branch, join, ratio = _measure(features)
    mm, conv, add, relu, reduce = (ratio[key] for key in ("MATMUL", "CONV", "ADD", "RELU", "REDUCE"))
    sub, div, mul, exp, sigmoid = (ratio[key] for key in ("SUB", "DIV", "MUL", "EXP", "SIGMOID"))

    # 规则按“最有先验区分度的 motif”排序，越靠前的越特化。
    # 1) 极窄超深 + Reduce/RELU/ADD 的链式网络（NARROW 类的唯一代表）。
    if width <= 16 and depth >= 100 and reduce >= 0.15 and relu >= 0.30 and add >= 0.30:
        pattern = GraphPattern.NARROW_DEEP_REDUCE_RELU_ADD
        reason = "层宽极窄且深度很大，Reduce/RELU/ADD 为主"
    # 2) 门控 MLP：MATMUL + SIGMOID + MUL 三者同时高占比，几乎只能来自门控结构。
    elif mm >= 0.25 and mul >= 0.15 and sigmoid >= 0.08:
        pattern = GraphPattern.GATED_SIGMOID_MLP
        reason = "MATMUL、SIGMOID 与 MUL 同时高占比，符合门控 motif"
    # 3) MatMul-Add 批处理：计算与加法几乎占全部，RELU 很少（说明没激活后处理）。
    elif mm >= 0.30 and add >= 0.30 and mm + add >= 0.68 and relu < 0.10:
        pattern = GraphPattern.WIDE_MATMUL_ADD
        reason = "MATMUL 与 ADD 绝对主导，RELU 很少"
    # 4) Attention/normalize：归一化链必备的 SUB/EXP/DIV 加 REDUCE，分支+汇合密集。
    elif sub + div + exp >= 0.16 and reduce >= 0.10 and branch >= 0.15 and join >= 0.30:
        pattern = GraphPattern.ATTENTION_NORMALIZE
        reason = "SUB/EXP/DIV/REDUCE 归一化链且分支、汇合密集"
    # 5) 浅层宽并行 compute-activation：拓扑是判别式；第二子句放宽到 32 层的
    #    较深 MATMUL-RELU 变体，与下面的残差块区分。
    # 部分生成式宽图把 MATMUL 替换成 CONV，所以这里以拓扑为准。
    elif depth <= 8 or (depth <= 32 and mm >= 0.20 and relu >= 0.20):
        pattern = GraphPattern.SHALLOW_WIDE_COMPUTE_ACTIVATION
        reason = "计算图很浅，存在大层宽的可并行 compute-activation 块"
    # 6) CNN/Residual：CONV/RELU/ADD（或残差等价形）几乎覆盖全图。
    elif conv + relu + add >= 0.95 and relu >= 0.30 and add >= 0.15:
        pattern = GraphPattern.CNN_RESIDUAL
        reason = "CONV/RELU/ADD（或其残差等价形）几乎覆盖全图"
    # 7) 兜底：中等深度的混合 MLP/Reduce 排布，不属于任何专属 motif。
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
    """对原始输入图做完整分析后返回七类分类报告。"""
    return classify_features(analyze_graph(graph))


def family_for_pattern(pattern: GraphPattern) -> GraphPatternFamily:
    """根据细粒度模式查询对应的粗粒度路由族。"""
    return _FAMILY_BY_PATTERN[pattern]


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：读取图 → 分类 → 输出或落盘 JSON 报告。"""
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
