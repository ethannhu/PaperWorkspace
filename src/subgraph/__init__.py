"""Public graph-analysis and algorithm entry points."""

from .demo_algorithm import analyze_graph, classify_op, describe_graph_pattern
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
