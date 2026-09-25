# MIXED 图策略

MIXED 图由 `graph_patterns.py` 识别为 `MIXED_MLP_REDUCE`，并路由到 `q2_algorithm._mixed_plan()`。Q2 和 Q3 都会经过这一分支；Q1 有自己的入口和边界代价策略。

## 当前实现

1. 调用 `semantic_partition()`，并开启 singleton repair。
2. 计算 Partition-DAG 上的一条重关键路径。
3. 使用 `_schedule_complex_partitions()` 调度，但把关键路径分裂惩罚缩放为 `0.35`，避免为保护一条路径牺牲过多分支并行度。
4. 图较宽或较深时使用 `max_ops=10`、`max_cycles=12000`；否则使用 `max_ops=14`、`max_cycles=18000`。

## 调度目标

调度器综合考虑分块周期和向上秩、已放置前驱带来的就绪时间、跨核边传输量、各 Core 的 Pipe 负载，以及关键路径前驱的 Core 亲和性。

算法输出仍只包含标准 `plan`。官方评估器负责补充 COPY、片上内存依赖和最终执行时间。
