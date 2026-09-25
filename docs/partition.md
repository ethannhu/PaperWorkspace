# 当前语义分块实现

实现文件是 `src/subgraph/semantic_partition.py`。它接收 `GraphFeatures`，返回 `Partition` 列表；不直接读取题目场景，也不生成 COPY。

## Partition 字段

- `id`：连续的子图 ID；
- `ops`：其中的非 COPY 操作 ID；
- `cycles`：操作周期之和；
- `rank_u`：原始 Op-DAG 的向上秩最大值；
- `pipe_cycles`：各 Pipe 的周期统计；
- `preds`、`succs`：分块后的 DAG 依赖。

## 实际流程

1. 根据 `op`、`pipe`、入度和出度提取计算、逐元素、归约、通信或其他角色。
2. 从拓扑顺序构造初始语义块。
3. 对相邻候选块计算合并分数，并受 `max_ops`、`max_cycles` 限制。
4. 拒绝会造成明显并行度损失、资源关系不匹配或违反拓扑结构的合并。
5. 按需执行 singleton repair，修复少量孤立单算子块。
6. 重新编号并建立 Partition-DAG。

## 当前合并倾向

代码优先考虑：

- `COMPUTE -> ELEMENTWISE`；
- `REDUCTION -> ELEMENTWISE/REDUCTION`；
- 明确的通信—计算阶段；
- 连续逐元素链和同 Pipe 计算块。

入度大于 1 或出度大于 1 的节点会产生边界惩罚。相邻块之间的中间 Tensor 复用和传输周期会增加合并收益；并行度损失、边界复杂度和连续计算串行化会降低收益。

## 调用参数

```python
semantic_partition(
    features,
    max_ops=16,
    max_cycles=20000,
    enable_singleton_repair=False,
)
```

Q1、wide、complex 和 mixed 分支会按图形状传入不同上限；这些是当前实现的启发式参数，不是题目固定参数。`buffer_pressure`、`pipelineability` 等字段仍保留在内部语义块结构中，但不是当前分块器的独立优化目标。

`ideas/` 中的 heavy-edge coarsening、显式 Cache reuse distance、复杂局部搜索等内容是候选方向；除非代码中明确实现，否则不属于当前算法。
