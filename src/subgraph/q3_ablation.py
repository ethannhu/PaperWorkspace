# ============================================================
# 人工智能工具信息 | AI Tool Information
#   工具名称 (Tool Name)        : GLM-5.2
#   版本/型号 (Version/Model)   : GLM-5.2
#   开发机构/公司 (Developer)    : 智谱AI (Zhipu AI / zai-org)
#   版本颁布日期 (Release Date) : 2026-06-13
#   声明：本程序及代码是在人工智能工具辅助下完成的
# ============================================================
"""Q3 五组消融变体的入口，统一转发到生产版的 Q3 实现。

消融变体说明：
    * ``build_full``                    —— 完整策略（无消融）；
    * ``build_no_l2``                   —— 关闭 L2 缓存复用；
    * ``build_no_cache_priority``       —— 关闭分支重图族的缓存优先级排序；
    * ``build_global_cache_priority``   —— 强制所有图族都启用缓存优先级；
    * ``build_no_communication``        —— 关闭跨核通信感知。
"""

from __future__ import annotations

from .algorithm_common import GraphFeatures
from .interfaces import AlgorithmResult
from .q3_algorithm import build_plan


def _make(name, graph, num_cores=4, features=None):
    """统一转发到 Q3 ``build_plan``，并通过 ``ablation`` 选择变体。"""
    return build_plan(graph, num_cores=num_cores, features=features, ablation=name)


def build_full(graph, num_cores=4, features: GraphFeatures | None = None) -> AlgorithmResult:
    """完整 Q3 策略，作为消融实验的对照基线。"""
    return build_plan(graph, num_cores=num_cores, features=features)


def build_no_l2(graph, num_cores=4, features=None):
    """关闭 L2 缓存复用，方案会写入 ``q3_cache_mode="disabled"``。"""
    return _make("no_l2", graph, num_cores, features)


def build_no_cache_priority(graph, num_cores=4, features=None):
    """关闭分支重图族的缓存优先级排序（仍保留 L2 复用本身）。"""
    return _make("no_cache_priority", graph, num_cores, features)


def build_global_cache_priority(graph, num_cores=4, features=None):
    """强制所有图族都启用缓存优先级排序（不只限原本的分支重 motif）。"""
    return _make("global_cache_priority", graph, num_cores, features)


def build_no_communication(graph, num_cores=4, features=None):
    """关闭跨核通信感知，调度只看核心负载。"""
    return _make("no_communication", graph, num_cores, features)


__all__ = ["build_full", "build_no_l2", "build_no_cache_priority",
           "build_global_cache_priority", "build_no_communication"]
