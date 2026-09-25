# MIXED 图算法设计

`case_graph_patterns.md` 中的 MIXED 类有 22 个样本，中位规模约为 1297 个
Op、34 层、最大层宽 152。它同时包含 MATMUL、RELU/ADD、REDUCE 和少量
SUB/DIV/MUL，分支不极端但汇合明显。因此它不适合把每个局部链都融合，也不
适合像超深链一样把整条路径固定在一个 Core 上。

## 算法

1. 在 COPY 收缩后的 Op-DAG 上提取语义角色：`COMPUTE`、`ELEMENTWISE`、
   `REDUCTION` 和 `COMMUNICATION`。
2. 用现有语义融合器构造候选块，优先融合 `COMPUTE -> ELEMENTWISE`、
   `REDUCTION -> ELEMENTWISE/REDUCTION` 和明确的流水阶段连接。分支、汇合、
   资源冲突和块大小限制阻止无条件融合。
3. 对融合后的 singleton 做两轮局部修复，仅接受能减少边界搬运且不制造环、
   不超过资源上限的合并。
4. 在 partition-DAG 上计算带通信估计的最长路径：

   ```text
   rank(P) = cycles(P) + max(ceil(bytes(P,Q)/60) + rank(Q))
   ```

   选择一条最重路径作为保护 spine。列表调度优先放置 spine，再按 upward
   rank 处理其余 ready 块；关键前驱、重汇合前驱和目标块尽量保持在同一
   Core，但只加软惩罚，以保留 MIXED 图的旁路并行度。

## 自适应参数

通常 MIXED 图使用 `max_ops=14`、`max_cycles=18000`。当层宽至少 180 或
深度至少 60 时，改用 `10` 和 `12000`，避免宽分支被融合成过大的任务。
关键路径粘性惩罚为复杂图策略的 35%，对应“保护串行瓶颈、不过度牺牲并行”
的折中。

算法输出仍只有标准的 `node_to_subgraph` 和 `core_schedules`；COPY、通信、
缓存驻留和最终 makespan 由官方 evaluator 计算。
