"""Minimal unified algorithm framework entry points."""

__all__ = [
    "analyze_graph", "build_plan", "build_partitions", "classify_op",
    "GraphPattern", "GraphPatternReport", "classify_features", "classify_graph",
]


def __getattr__(name: str):
    # Keep ``python -m subgraph.demo_framework`` free of an eager-import warning.
    if name in __all__:
        if name in {"GraphPattern", "GraphPatternReport", "classify_features", "classify_graph"}:
            from . import graph_patterns
            return getattr(graph_patterns, name)
        from . import demo_framework
        return getattr(demo_framework, name)
    raise AttributeError(name)


def main() -> None:
    from .demo_framework import main as framework_main

    raise SystemExit(framework_main())
