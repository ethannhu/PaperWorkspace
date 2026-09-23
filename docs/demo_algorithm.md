# Demo 算法统一设计

本文是 `docs/demo_model.md`、`docs/demo_solution.md` 和
`docs/demo_code_arch.md` 的整合版。它定义 demo 算法的统一术语、三道题的
约束差异、算法流程和可实现的输出边界；其他文档分别保留模型推导、方案讨论
和工程细节，不再另行定义题目规则。

## 1. 先固定问题边界

参赛算法的输出只有三部分决策：

```text
非 COPY Op → 子图 sgid
子图 sgid → Core
同一 Core 上的子图顺序
```

算法不直接输出单个 Op 的执行时刻，也不手工生成 `COPY_IN`、`COPY_OUT`。
跨子图/跨核搬运、核内 Op 调度、Pipe 并行、L1/UB 换入换出、DDR 带宽竞争
和问题 3 的 Cache 行为由官方评估器完成。因此，静态公式只能用于生成和筛选
候选方案，最终性能必须以官方 evaluator 为准。

标准方案格式为：

```json
{
  "node_to_subgraph": {"11": 0, "12": 0, "20": 1},
  "core_schedules": [[0], [1], []]
}
```

约束是：所有且仅所有非 COPY Op 必须映射一次；每个 `sgid` 必须在
`core_schedules` 中出现一次；子图依赖图和每个 Core 上的顺序都必须保持无环。

固定硬件参数以 `docs/challenge_data_contract.md` 和
`docs/challenge_lookup.md` 为准：

| 项目 | Q1 | Q2 | Q3 |
|---|---:|---:|---:|
| 场景 | A | B | B + 只读 L2 |
| L1 / UB | 512 KiB / 128 KiB | 同左 | 同左 |
| DDR 总带宽 | 60 B/cycle | 60 B/cycle | 60 B/cycle |
| 跨核等待/同步 | 1000 cycles | 500 cycles COPY 延迟 | 500 cycles COPY 延迟 |
| L2 | 无 | 无 | 1 MiB，250 B/cycle，FIFO |

## 2. 统一抽象

原始计算图是带 Tensor 的 Op-DAG。对每个 Op (v) 提取：

$$
f_v=(cycles_v, pipe_v, type_v, inputs_v, outputs_v, rank_v)
$$

对每个 Tensor (x) 提取：

$$
g_x=(size_x, pos_x, fanout_x, producer_x, consumers_x).
$$

其中 `pipe` 直接使用输入 JSON 的值；不能仅根据 Op 名称推断流水线。
COPY Op 不参与 `node_to_subgraph`，也不参与自定义子图划分。

统一优化链为：

$$
\text{特征提取}
\rightarrow\text{初始划分}
\rightarrow\text{子图图构建}
\rightarrow\text{分核与排序}
\rightarrow\text{局部改进}
\rightarrow\text{官方评估}
$$

三道题不更换算法骨架，只更换通信和存储代价模型。

## 3. 三道题的不可混用规则

### 3.1 Q1：场景 A，跨子图必经 DDR

Q1 中：

```text
一个子图 = 一个独立 Task
```

只要两个 Op 属于不同子图，无论两个子图是否位于同一 Core，它们之间的
数据都必须经过：

```text
源子图 → COPY_OUT → DDR → COPY_IN → 目标子图
```

因此 Q1 的通信惩罚是 **跨子图边**，不是跨核边：

$$
D_1=\sum_{(u,v)\in E}size(u,v)\,I[sg(u)\ne sg(v)].
$$

同核相邻 Task 的切换等待为 100 cycles，跨核前驱等待为 1000 cycles；所有
DDR 搬运共享 60 B/cycle。Q1 的核心权衡是：

$$
\text{并行度}\quad\leftrightarrow\quad\text{跨子图 DDR 搬运与 Task 开销}.
$$

### 3.2 Q2：场景 B，同核子图可驻留

Q2 中，同一 Core 上的全部子图合并为一个 Task。两个不同子图若位于同一
Core，其依赖数据可以留在 L1/UB，不插入边界 COPY；只有跨 Core 的依赖
需要 `COPY_OUT/COPY_IN`，并承担 500 cycles 的跨核 COPY 延迟。

因此 Q2 的通信惩罚是 **跨核边**：

$$
D_2=\sum_{(u,v)\in E}size(u,v)\,I[core(u)\ne core(v)].
$$

Q2 还要考虑核内 Tensor 驻留造成的压力，但不应把一个静态的“缓存峰值”当作
最终执行时间。官方核内调度器会根据真实顺序处理驻留、换出和换入；算法可用
live-range、峰值驻留量和预估 spill 作为候选方案的代理指标。

### 3.3 Q3：Q2 + 共享只读 FIFO L2

Q3 完全继承 Q2 的 Task 和跨核规则，只改变 `COPY_IN` 的数据源：

- 命中：从共享只读 L2 读取，使用 250 B/cycle；命中不刷新 FIFO 顺序；
- 未命中：从 DDR 读取，使用 DDR 带宽，完成后写入 L2；
- 容量不足时按 FIFO 淘汰；单个 Tensor 大于 1 MiB 时不缓存。

算法不能直接输出“把哪个 Tensor 放进 L2”。因此 `fanout × size`、关键路径
和近似 reuse distance 只能作为分核/排序的启发式；真实命中率仍由官方评估器
决定。Q3 的新权衡是：减少重复 DDR 读取，还是利用更低的共享输入成本换取
更多跨核并行。

## 4. 统一算法流程

### 4.1 特征提取

按拓扑序计算：

- Op 的 `cycles`、`pipe`、语义类型和输入输出 Tensor；
- DAG depth 和 upward rank；
- 分支/汇聚点、重复结构和大 Tensor 边；
- Tensor 的大小、fanout、最后一次使用位置和跨核复用潜力。

upward rank 可写成：

$$
rank(v)=cycles(v)+\max_{w\in succ(v)}
\left(comm(v,w)+rank(w)\right),
$$

出口节点的 rank 等于自身周期。这里的 `comm` 是当前问题的代理成本：Q1
按跨子图估计，Q2/Q3 按跨核估计，不能把三种情形混成一个固定常数。

### 4.2 初始子图划分

先采用稳定的语义规则生成可行初值：沿单一路径吸收适合融合的 elementwise
Op；遇到分支、汇聚、REDUCE、资源阈值或明显的并行结构时停止。随后对相邻
子图计算合并收益：

$$
Gain(P_i,P_j)=
\alpha\,locality
+\beta\,pipeline
-\gamma\,parallel\_loss
-\delta\,communication.
$$

Q1 的 `communication` 对应跨子图 DDR；Q2/Q3 更关注未来跨核搬运和缓存压力。
不要把“一个 Op 一个子图”作为默认策略，也不要无条件把整条 DAG 合并成一个
大子图。

### 4.3 初始分核与排序

在 Partition-DAG 上使用关键路径优先的列表调度：从 ready 子图中优先选择
rank 最大者，对每个 Core 估计最早完成时间：

$$
EFT(P,c)=EST(P,c)+T(P).
$$

选择代理 EFT 最小的 Core，并保留依赖合法的同核顺序。Q1 的 `EST` 加入
场景 A 的 100/1000 cycles 和跨子图搬运；Q2/Q3 加入跨核 COPY 延迟，并将
同核依赖视为可驻留而非自动 DDR 通信。

### 4.4 局部改进

以当前最慢 Core、关键路径和大搬运边为重点，尝试以下邻域：

1. 合并相邻子图；
2. 拆分缓存压力过大的子图；
3. 把子图迁移到另一个 Core；
4. 交换两个 Core 的子图；
5. 调整同一 Core 上的合法顺序。

先用代理代价筛选候选，再调用官方 evaluator 复核。接受使 Makespan 改善的
方案；若并列，再比较额外搬运量和 Q3 的 Cache 命中率。局部搜索必须有时间
预算，不能暴力枚举全部划分。

## 5. 统一目标与实现边界

可以用一个场景参数化的代理目标表示三道题：

$$
J_s=\alpha T_{proxy}
+\beta D_s
+\gamma Spill_s
+\eta Imbalance_s
-\mu Reuse_s,
$$

其中：

- Q1：(D_s=D_1)，`Reuse` 关闭；
- Q2：(D_s=D_2)，启用 L1/UB 驻留和 spill 代理；
- Q3：在 Q2 基础上启用 L2 reuse 代理。

该目标不是官方评分公式，也不是要求实现一个 MILP。它只服务于初值、排序
和局部搜索；最终指标以 evaluator 输出的 Makespan、额外搬运量和 Q3 Cache
命中率为准。

工程上建议分三阶段：

```text
V1  解析 JSON → 合法划分/分核/排序 → 输出标准 JSON
V2  加入 rank、通信代理、语义合并和局部 move/merge/split
V3  加入 live-range、spill 代理、L2 reuse 代理和 evaluator-in-the-loop
```

## 6. 文档使用顺序

| 文档 | 用途 | 权威范围 |
|---|---|---|
| `challenge_data_contract.md` | JSON、配置、评估接口 | 实现接口和固定参数 |
| `challenge_lookup.md` | 题面事实和评估器行为 | 题目规则 |
| `demo_algorithm.md` | 本文，统一算法总纲 | 算法术语和三问边界 |
| `demo_model.md` | 数学模型展开 | 代理目标和论文叙事 |
| `demo_solution.md` | 比赛实现路线 | baseline 和局部搜索取舍 |
| `demo_code_arch.md` | Python 工程拆分 | 模块、数据结构和代码骨架 |

若模型推导、方案建议或工程草图与题目规则冲突，以 `challenge_*` 文档和
官方 evaluator 为准；若三个 demo 文档的算法表述冲突，以本文的 Q1/Q2/Q3
定义和“输出边界”为准。
