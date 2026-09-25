# Algorithm entry points

The planning pipeline exposes one algorithm entry point:

```python
from subgraph.q2_algorithm import build_plan

result = build_plan(graph, num_cores=4)
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
partitions.  New work should replace a whole family strategy rather than
mixing an unrelated partitioner with an unrelated scheduler.

The complex family is the first specialized branch.  It targets
CNN/Residual and Attention/Normalize graphs by combining semantic operator
fusion with critical-path protection:

- run semantic partitioning with singleton repair and slightly larger block
  limits, so isolated residual or normalize-chain operators are not left as
  unnecessary one-op tasks;
- find the heaviest critical path in the partition DAG;
- schedule most complex graphs with a sticky critical spine, keeping the
  protected path and expensive joins on the predecessor core when that avoids
  costly cross-core traffic;
- for very long CNN residual spines, keep the complex fusion but fall back to
  plain list scheduling to preserve convolution-level parallelism.

The batch evaluator selects the question-specific entry points directly:

```text
subgraph.q1_algorithm:build_plan
subgraph.q2_algorithm:build_plan
subgraph.q3_algorithm:build_plan
```

Diagnostics contain the graph-pattern report and the selected algorithm
strategy name, which is enough to audit pattern routing during benchmarks.
