# 算法框架与入口

算法只负责根据原始计算图生成切图和核调度方案，不负责生成 `COPY`、核内换入换出或模拟最终执行；这些工作由 `artifacts/code/` 中的官方评估器完成。

## 统一流程

```text
原始图 -> analyze_graph -> classify_features -> family strategy -> AlgorithmResult
```

`analyze_graph()` 位于 `src/subgraph/algorithm_common.py`。它收缩 COPY 节点并建立可调度的 Op-DAG。算法只对非 `COPY_IN`、非 `COPY_OUT` 操作分块。

## 三个入口

```text
subgraph.q1_algorithm:build_plan
subgraph.q2_algorithm:build_plan
subgraph.q3_algorithm:build_plan
```

三个入口都返回 `AlgorithmResult`：

```python
result.plan         # node_to_subgraph、core_schedules
result.diagnostics  # 图模式和策略诊断
```

### Q1

Q1 使用较大的语义块和更强的边界规避策略。由于同核子图边界也有固定等待，Q1 的调度使用 `q1` 通信代价，并对关键路径保持较强的 Core 亲和性。

### Q2

Q2 使用 `q2` 通信模型：同核子图之间不产生跨核 COPY，跨核依赖使用固定延迟和传输时间。代码按图形状选择 wide、complex、mixed 或普通 semantic 分支，并使用关键路径、边界搬运量和 Pipe 负载进行列表调度。

### Q3

Q3 复用 Q2 的图模式和调度策略，但明确以 `scenario="q3"` 构造方案，并在诊断中标记只读 FIFO Cache 模型。Q3 的 Cache 命中、容量淘汰和 Cache 带宽由官方问题 3 评估器处理，算法不自行模拟 Cache。

## 当前策略分支

`src/subgraph/q2_algorithm.py` 中的粗粒度分支如下：

- `wide`：语义块合并、拓扑区间合并，部分形状使用流水式拓扑分配；
- `complex`：语义融合、singleton 修复和关键路径 sticky 调度；
- `mixed`：较小分块、singleton 修复和较弱的关键路径粘性；
- 其他形状：普通语义分块和通信感知列表调度。

`q1_algorithm.py` 在此基础上使用 Q1 专属的边界参数。`q3_algorithm.py` 是 Q3 的公开入口，不是另一个独立的事件模拟器。

## 运行测试

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
