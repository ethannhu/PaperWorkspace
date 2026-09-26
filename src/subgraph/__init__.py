# ============================================================
# 人工智能工具信息 | AI Tool Information
#   工具名称 (Tool Name)        : GLM-5.2
#   版本/型号 (Version/Model)   : GLM-5.2
#   开发机构/公司 (Developer)    : 智谱AI (Zhipu AI / zai-org)
#   版本颁布日期 (Release Date) : 2026-06-13
#   声明：本程序及代码是在人工智能工具辅助下完成的
# ============================================================
"""子图算法包的对外入口。

汇总公开的图分析、模式识别和主入口函数：
    * ``analyze_graph``          —— COPY 收缩 + 拓扑分析，得到 ``GraphFeatures``；
    * ``classify_op``            —— 把单条 op 标签成 DENSE_COMPUTE/REDUCTION/...；
    * ``describe_graph_pattern`` —— 输出当前图的七类模式诊断字典；
    * 七类模式枚举与 ``classify_features`` / ``classify_graph``；
    * ``build_plan``             —— 默认实现指向 **Q2** 的 ``build_plan``。

注意：``build_plan`` 默认绑定到 Q2 的实现；如果需要 Q1 或 Q3 的入口，需
要显式从 ``subgraph.q1_algorithm`` / ``subgraph.q3_algorithm`` 导入。
"""

from .algorithm_common import analyze_graph, classify_op, describe_graph_pattern
from .graph_patterns import (
    GraphPattern,
    GraphPatternFamily,
    GraphPatternReport,
    classify_features,
    classify_graph,
    family_for_pattern,
)
from .q2_algorithm import build_plan

__all__ = [
    "analyze_graph", "build_plan", "classify_op", "describe_graph_pattern",
    "GraphPattern", "GraphPatternFamily", "GraphPatternReport",
    "classify_features", "classify_graph", "family_for_pattern",
]
