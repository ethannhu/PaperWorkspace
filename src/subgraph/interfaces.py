# ============================================================
# 人工智能工具信息 | AI Tool Information
#   工具名称 (Tool Name)        : GLM-5.2
#   版本/型号 (Version/Model)   : GLM-5.2
#   开发机构/公司 (Developer)    : 智谱AI (Zhipu AI / zai-org)
#   版本颁布日期 (Release Date) : 2026-06-13
#   声明：本程序及代码是在人工智能工具辅助下完成的
# ============================================================
"""算法与评测工具共享的轻量结果对象。

``AlgorithmResult`` 是算法模块（Q1/Q2/Q3）与外部评测器之间唯一的契约：
``plan`` 字段是提交给官方评测器的方案（含 ``node_to_subgraph`` 与
``core_schedules``），``diagnostics`` 字段记录算法内部决策，便于后续分析。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AlgorithmResult:
    """算法返回给评测器的结果对象。

    使用 ``frozen=True`` 让实例不可变：算法实现不能在返回后偷偷修改计划，
    评测器可以放心引用同一个对象。
    """

    plan: dict[str, Any]
    diagnostics: dict[str, Any]


__all__ = ["AlgorithmResult"]
