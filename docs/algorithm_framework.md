# Algorithm framework

The planning pipeline exposes one algorithm entry point:

```python
from subgraph.demo_framework import build_plan

result = build_plan(graph, num_cores=4, scenario="q2")
plan = result.plan
diagnostics = result.diagnostics
```

`build_plan` does four things:

1. `analyze_graph(graph)` builds the COPY-contracted op DAG.
2. `classify_features(features)` assigns the graph to one fine pattern and one
   coarse family: `wide`, `narrow`, `mixed`, or `complex`.
3. The selected family strategy owns both partitioning and scheduling.
4. The result is returned as `AlgorithmResult(plan, diagnostics)`.

Scheduling is not a replaceable framework stage.  It is part of the selected
strategy because placement choices depend on how that strategy formed
partitions.  The current four family branches all call the same semantic
strategy internally; later work should replace a whole family strategy rather
than mixing an unrelated partitioner with an unrelated scheduler.

The command-line entry point mirrors the same simple path:

```text
python -m subgraph.demo_framework graph.json -n 4 --scenario q2 -o plan.json
```

Diagnostics contain the graph-pattern report and the selected algorithm
strategy name, which is enough to audit pattern routing during benchmarks.
