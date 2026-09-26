# ============================================================
# 人工智能工具信息 | AI Tool Information
#   工具名称 (Tool Name)        : GLM-5.2
#   版本/型号 (Version/Model)   : GLM-5.2
#   开发机构/公司 (Developer)    : 智谱AI (Zhipu AI / zai-org)
#   版本颁布日期 (Release Date) : 2026-06-13
#   声明：本程序及代码是在人工智能工具辅助下完成的
# ============================================================
"""Q2 五组消融变体的入口，统一转发到生产版的 Q2 实现。

消融变体说明：
    * ``build_full``              —— 完整策略（无消融）；
    * ``build_no_structure``      —— 关闭图族路由与 motif 分支，仅走默认语义分区；
    * ``build_no_communication``  —— 关闭跨核通信感知，调度只看负载；
    * ``build_no_critical_path``  —— 关闭关键路径粘性调度；
    * ``build_no_core_control``   —— 关闭窄残差链的自适应核数控制。
"""

from __future__ import annotations

from .algorithm_common import GraphFeatures
from .interfaces import AlgorithmResult
from .q2_algorithm import build_plan


def _make(name, graph, num_cores=4, features=None):
    """统一转发到 Q2 ``build_plan``，并通过 ``ablation`` 选择变体。"""
    return build_plan(graph, num_cores=num_cores, features=features, ablation=name)


def build_full(graph, num_cores=4, features: GraphFeatures | None = None) -> AlgorithmResult:
    """完整 Q2 策略，作为消融实验的对照基线。"""
    return build_plan(graph, num_cores=num_cores, features=features)


def build_no_structure(graph, num_cores=4, features=None):
    """关闭图族路由与 motif 分支，走默认语义分区。"""
    return _make("no_structure", graph, num_cores, features)


def build_no_communication(graph, num_cores=4, features=None):
    """关闭跨核通信感知，调度只看核心负载。"""
    return _make("no_communication", graph, num_cores, features)


def build_no_critical_path(graph, num_cores=4, features=None):
    """关闭关键路径粘性调度，所有分区走普通列表调度。"""
    return _make("no_critical_path", graph, num_cores, features)


def build_no_core_control(graph, num_cores=4, features=None):
    """关闭窄残差链场景下的自适应核数控制（即不限制有效核数）。"""
    return _make("no_core_control", graph, num_cores, features)


__all__ = ["build_full", "build_no_structure", "build_no_communication",
           "build_no_critical_path", "build_no_core_control"]
