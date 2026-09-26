# ============================================================
# 人工智能工具信息 | AI Tool Information
#   工具名称 (Tool Name)        : GLM-5.2
#   版本/型号 (Version/Model)   : GLM-5.2
#   开发机构/公司 (Developer)    : 智谱AI (Zhipu AI / zai-org)
#   版本颁布日期 (Release Date) : 2026-06-13
#   声明：本程序及代码是在人工智能工具辅助下完成的
# ============================================================
"""Q1 五组消融变体的可调用入口，供批量实验脚本统一调用。

每个 ``build_*`` 函数最终都转发到 ``q1_algorithm.build_plan``，仅在调用时
传入不同的 ``ablation=`` 名称。这样批量评测器只需识别同一组入口符号即可，
消融开关与算法逻辑仍然集中在 ``q1_algorithm`` 中维护。

消融变体说明：
    * ``build_full``              —— 完整策略（无消融）；
    * ``build_no_semantic``       —— 关闭语义分区，仅按拓扑顺序切块；
    * ``build_no_adaptive``       —— 关闭按图族自适应粒度，使用固定大小；
    * ``build_no_critical_path``  —— 关闭关键路径粘性调度；
    * ``build_no_rebalance``      —— 关闭全局再平衡迁移阶段。
"""

from __future__ import annotations

from .algorithm_common import GraphFeatures
from .interfaces import AlgorithmResult
from .q1_algorithm import build_plan


def _variant(name: str, graph: dict, num_cores: int = 4,
             features: GraphFeatures | None = None) -> AlgorithmResult:
    """统一转发到 Q1 的 ``build_plan``，并通过 ``ablation`` 选择变体。"""
    return build_plan(graph, num_cores=num_cores, features=features, ablation=name)


def build_full(graph: dict, num_cores: int = 4,
               features: GraphFeatures | None = None) -> AlgorithmResult:
    """完整 Q1 策略，作为消融实验的对照基线。"""
    return build_plan(graph, num_cores=num_cores, features=features)


def build_no_semantic(graph: dict, num_cores: int = 4,
                      features: GraphFeatures | None = None) -> AlgorithmResult:
    """A1：禁用语义分区，按拓扑顺序与固定大小切块。"""
    return _variant("no_semantic", graph, num_cores, features)


def build_no_adaptive(graph: dict, num_cores: int = 4,
                      features: GraphFeatures | None = None) -> AlgorithmResult:
    """A2：禁用按图族自适应的分区大小，统一使用固定的 24 ops / 32000 cycles。"""
    return _variant("no_adaptive", graph, num_cores, features)


def build_no_critical_path(graph: dict, num_cores: int = 4,
                           features: GraphFeatures | None = None) -> AlgorithmResult:
    """A3：禁用关键路径粘性调度，所有分区走普通列表调度。"""
    return _variant("no_critical_path", graph, num_cores, features)


def build_no_rebalance(graph: dict, num_cores: int = 4,
                       features: GraphFeatures | None = None) -> AlgorithmResult:
    """A4：禁用全局再平衡迁移阶段，保留调度器输出的初始核分配。"""
    return _variant("no_rebalance", graph, num_cores, features)


__all__ = [
    "build_full", "build_no_semantic", "build_no_adaptive",
    "build_no_critical_path", "build_no_rebalance",
]
