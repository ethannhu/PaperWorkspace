# Algorithm framework

The planning pipeline is intentionally small and uses callable injection:

```python
from subgraph.demo_framework import build_plan
from subgraph.naive_partition import naive_partition

result = build_plan(graph, partitioner=naive_partition)
plan = result.plan
diagnostics = result.diagnostics
```

`build_plan` has three stages:

1. `analyze_graph(graph)` produces `GraphFeatures`.
2. `partitioner(features, max_ops, max_cycles)` produces `list[Partition]`.
3. `scheduler(partitions, features, num_cores, scenario, return_diagnostics=True)`
   produces core placement, per-core order, and scheduler diagnostics.

An optional `schedule_optimizer` can refine the scheduler result. The complete
algorithm result is an `AlgorithmResult` containing `plan` and `diagnostics`.
The evaluation framework writes the collected diagnostics to
`case_xxx/diagnostics.json`; new experiments should provide a callable for only
the stage they change. There is no command-line diagnostics path or
algorithm-name dispatch in the pipeline.

The command-line entry point accepts the same stages as `module:callable`:

```text
python -m subgraph.demo_framework graph.json \
  --partitioner my_experiment:partition \
  --scheduler my_experiment:schedule
```
