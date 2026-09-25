"""Small result object shared by algorithms and evaluation tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AlgorithmResult:
    """The algorithm result passed from an algorithm to the evaluator."""

    plan: dict[str, Any]
    diagnostics: dict[str, Any]


__all__ = ["AlgorithmResult"]
