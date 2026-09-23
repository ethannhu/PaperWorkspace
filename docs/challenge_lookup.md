# 2026 年中国研究生数学建模竞赛 A 题题面速查

> 题目：**通用神经网络处理器下的多核调度问题**  
> 用途：供后续建模 / 编程 Agent 直接读取，避免反复解析原始 `challenge.docx`。  
> 原则：仅整理题目事实、固定参数、输入输出与评估器规则；不包含具体解题方案。

---

## 1. 题目核心

给定一张神经网络计算 DAG 和 `K` 个同构 NPU 核心，需要联合决定：

1. **切图**：每个非 `COPY_IN/COPY_OUT` 操作属于哪个子图 `sgid`；
2. **核心分配**：每个子图放到哪个 NPU 核心；
3. **核内子图顺序**：同一核心上的多个子图按什么顺序执行。

目标：

- **首要目标：最小化总体任务执行时间 `Makespan`**；
- **次要目标：减少新增 DDR 数据搬运量**；
- 问题 3 还关注 **L2 Cache 命中率**。

算法需适应 `2~5` 核；正文实验要求报告 `1~5` 核结果。

---

# 2. 计算图定义

## 2.1 图结构

计算图是一个有向无环图 DAG，实际上是 **Tensor–Op 二部图**。

记：

- `T`：Tensor 节点集合；
- `O`：Op 节点集合；
- `V = T ∪ O`：全部节点；
- `E`：有向边集合；
- `K`：NPU 核心数量；
- `sgid`：子图编号，非负整数。

边只允许：

```text
Tensor -> Op
Op -> Tensor
```

不允许：

```text
Tensor -> Tensor
Op -> Op
```

每个节点有全局唯一整数 `id`。

---

## 2.2 Tensor 节点

JSON 字段：

```json
{
  "id": 1,
  "pos": "DDR | L1 | UB",
  "size": 1024
}
```

含义：

| 字段 | 含义 |
|---|---|
| `id` | Tensor 唯一编号 |
| `pos` | 逻辑存储位置：`DDR`、`L1` 或 `UB` |
| `size` | Tensor 大小，单位 byte |

规则：

- 计算图输入、最终输出 Tensor 位于 `DDR`；
- 中间 Tensor 位于 `L1` 或 `UB`；
- Tensor 大小既用于缓存容量判断，也用于数据搬运量计算。

---

## 2.3 Op 节点

JSON 字段：

```json
{
  "id": 11,
  "op": "ADD",
  "pipe": "PIPE_V",
  "cycles": 4
}
```

含义：

| 字段 | 含义 |
|---|---|
| `id` | Op 唯一编号 |
| `op` | 操作类型 |
| `pipe` | 使用的硬件 Pipe |
| `cycles` | 核内计算操作的执行周期数 |

典型操作：

- `COPY_IN`：DDR → L1/UB；
- `COPY_OUT`：L1/UB → DDR；
- `ADD`、`MUL` 等：核内计算操作。

注意：

- 核内计算操作的耗时由 `cycles` 给定；
- 访问 DDR 的 COPY 最终耗时**不直接按固定 cycles 结算**，而由搬运数据量及运行时共享带宽竞争决定。

---

## 2.4 Edge 节点

JSON 字段：

```json
{
  "source": 1,
  "target": 11
}
```

边本身不记录通信量。

若依赖跨越需要通信的 Task 边界，则搬运字节数由相应 Tensor 的 `size` 决定。

---

# 3. NPU 硬件模型

## 3.1 全局结构

平台包含：

- `K` 个**同构 AI Core**；
- 所有核心共享一个核外 `DDR`；
- 每个核心具有私有 `L1`、`UB`；
- 每个核心有多条独立 Pipe。

所有核心访问 DDR 时，共享固定总物理带宽。

---

## 3.2 每核资源

每个 NPU 核心包含：

| Pipe | 功能 |
|---|---|
| `PIPE_M` | Cube / 矩阵计算单元 |
| `PIPE_V` | Vector / 向量计算单元 |
| `PIPE_MTE2` | 数据搬运 |
| `PIPE_MTE3` | 数据搬运 |

其中：

- `PIPE_M` 和 `PIPE_V` 可并行；
- 同一个 Pipe 内操作串行；
- 不同 Pipe 可以并行，但仍受数据依赖、存储容量等约束。

### 搬运 Pipe

题面规定：

- `PIPE_MTE2`：支持 DDR → L1/UB，以及 UB → L1；
- `PIPE_MTE3`：支持 L1/UB → DDR，以及 L1 → UB。

L1 与 UB 间 COPY 只占对应 Pipe，不进入 DDR 总带宽池。

---

## 3.3 固定硬件参数

| 参数 | 固定值 |
|---|---:|
| L1 容量 | `524288 bytes` = 512 KiB |
| UB 容量 | `131072 bytes` = 128 KiB |
| DDR 总带宽 | `60 bytes/cycle` |
| 问题 1 跨核前驱等待 | `1000 cycles` |
| 问题 1 同核 Task 切换等待 | `100 cycles` |
| 问题 2/3 跨核 COPY 同步延迟 | `500 cycles` |
| 问题 3 L2 容量 | `1048576 bytes` = 1 MiB |
| 问题 3 L2 读带宽 | `250 bytes/cycle` |

DDR 读写共享同一个 `60 bytes/cycle` 带宽池。

---

# 4. 切图与调度结果格式

参赛算法需输出：

```text
<case>_multicore_res.json
```

顶层只能包含：

```json
{
  "node_to_subgraph": {},
  "core_schedules": []
}
```

---

## 4.1 `node_to_subgraph`

示例：

```json
{
  "11": 0,
  "15": 0,
  "20": 1
}
```

含义：

```text
非 COPY Op id -> sgid
```

要求：

- 必须覆盖**所有且仅所有非 `COPY_IN/COPY_OUT` Op**；
- 每个值为非负整数 `sgid`；
- 由跨子图依赖形成的子图依赖图必须仍然是 DAG。

---

## 4.2 `core_schedules`

示例，4 核：

```json
[
  [0, 3],
  [1],
  [2, 5],
  [4]
]
```

含义：

- 外层下标 = `core id`；
- 每个内层列表 = 该核心上子图执行顺序。

要求：

- 每个 `sgid` 必须且只能出现一次；
- 某核心可以没有子图，对应 `[]`；
- 同核顺序不能违反子图依赖关系。

---

# 5. 评估指标

## 5.1 Makespan

最重要指标。

定义：

```text
所有操作完成时刻的最大值
```

单位：

```text
cycle
```

最终 Makespan 由官方多核离散事件模拟器统一计算。

它考虑：

- 子图 / Task 依赖；
- 多核并行；
- 同核 Pipe 串行；
- 不同 Pipe 并行；
- L1/UB 容量；
- 缓存换入换出；
- 跨核同步延迟；
- DDR 带宽竞争；
- 问题 3 的 L2 Cache。

---

## 5.2 总额外数据搬运量

定义：

```text
scheduled_copy_bytes - original_graph_copy_bytes
```

新增搬运主要来自：

1. 跨 Task 边界数据搬运；
2. 多个 Task 对同一输入的重复读取；
3. L1/UB 容量不足导致的换出 + 再换入。

该指标为次要指标，但对 Makespan 也可能有显著影响，因为所有 DDR 搬运共享带宽。

---

## 5.3 Cache 命中率

仅问题 3 重点使用。

定义：

```text
可由只读 L2 Cache 服务的 COPY_IN 中：
命中字节数 / 总访问字节数
```

---

# 6. 场景 A 与场景 B

这是三道题之间最重要的区别。

## 6.1 场景 A

特点：

```text
1 个子图 = 1 个独立 Task
```

即使两个子图：

- 位于同一个核心；
- 前后连续执行；

它们之间的跨子图数据也**不能直接保留在私有缓存中**。

所有跨子图数据必须经过：

```text
源子图 -> COPY_OUT -> DDR -> COPY_IN -> 目标子图
```

并且同一核执行完一个 Task 后，Task 间缓存状态视为清空。

问题 1 使用场景 A。

---

## 6.2 场景 B

特点：

```text
同一个核心上的全部子图被合并为一个 Task
```

因此：

### 同核子图之间

可以直接复用：

```text
L1 / UB
```

不插入边界 `COPY_OUT/COPY_IN`。

### 跨核依赖

必须：

```text
源核 COPY_OUT
    ↓
DDR
    ↓
500 cycles 同步延迟
    ↓
目标核 COPY_IN
```

问题 2、3 使用场景 B。

---

# 7. 问题 1：场景 A 下的多核切图与调度

## 7.1 任务描述

在场景 A 下：

- 每个子图单独构成一个 Task；
- 所有跨子图数据均经 DDR 中转。

需要决定：

1. 每个非 COPY Op 的 `sgid`；
2. 各子图分配到哪个核心；
3. 每个核心上的子图执行顺序。

---

## 7.2 目标

主要：

```text
最小化 Makespan
```

同时：

```text
减少总额外 DDR 搬运量
```

要求算法在 `2~5` 核下稳定、高效地产生高质量结果。

---

## 7.3 问题 1 特有时序规则

### 同核相邻 Task

前一个 Task 完成后：

```text
至少等待 100 cycles
```

才能激活下一个 Task。

### 跨核前驱

跨核前驱 Task 完成后：

```text
至少等待 1000 cycles
```

目标 Task 才满足对应激活条件。

Task 实际激活时间取所有约束中的最大值。

---

## 7.4 结果要求

正文：

- 给出 `1~5` 核平均加速比折线图。

对于 `K > 1`：

```text
speedup(K)
= 单核整图核内调度 Makespan
  / K 核多核 Makespan
```

注意：

- 单核基准由官方启发式核内调度器固定给出；
- 因单核基准本身不保证最优，多核加速比可能大于核数；
- 多用例平均加速比应为**各用例 speedup 的算术平均**，不是总时间之比。

附录逐用例报告：

- Makespan；
- 总额外数据搬运量。

---

# 8. 问题 2：场景 B 下的多核切图与调度

## 8.1 任务描述

问题 2 切换到场景 B。

同一核心上的所有子图：

```text
合并为一个 Task
```

所以：

- 同核依赖直接保留在 L1/UB；
- 只有跨核依赖需要经过 DDR。

仍需给出：

```text
node_to_subgraph
core_schedules
```

---

## 8.2 目标

在严格满足：

```text
L1 / UB 容量约束
```

的前提下：

1. 缩短 Makespan；
2. 减少额外数据搬运。

要求适应 `2~5` 核。

---

## 8.3 问题 2 跨核通信规则

若数据从核心 A 传到核心 B：

```text
A: COPY_OUT
      ↓
     DDR
      ↓
   等待 500 cycles
      ↓
B: COPY_IN
```

目标 `COPY_IN` 只有在：

```text
源 COPY_OUT 完成 + 500 cycles
```

之后，才能进入 ready 判定。

---

## 8.4 与问题 1 的根本区别

问题 1：

```text
同核不同子图
也必须经过 DDR
```

问题 2：

```text
同核不同子图
可直接复用 L1 / UB
```

因此问题 2 中：

```text
“子图属于哪个核心”
```

会直接决定大量中间 Tensor 是否需要经过 DDR。

---

## 8.5 结果要求

正文：

- 场景 B 下 `1~5` 核平均加速比折线图。

附录逐用例：

- Makespan；
- 总额外数据搬运量。

使用与问题 1 相同的单核基准和加速比定义。

---

# 9. 问题 3：共享只读 L2 Cache 下的多核切图与调度

## 9.1 任务描述

问题 3 在问题 2 / 场景 B 的基础上增加：

```text
所有核心共享的只读 L2 Cache
```

参数：

```text
容量：1 MiB
带宽：250 bytes/cycle
```

L2 带宽与 DDR 带宽：

```text
相互独立
互不占用
```

其主要用途是：

```text
复用多个核心对共享输入 Tensor 的读取
```

---

## 9.2 Task 与通信规则

Task 结构完全沿用问题 2：

```text
同一核心上的全部子图 = 一个 Task
```

仍然只有跨核数据边需要插入 `COPY_OUT/COPY_IN`。

额外变化只发生在 `COPY_IN` 访问源上。

---

## 9.3 L2 Cache 行为

L2 是：

```text
只读 FIFO Cache
```

在 `COPY_IN` 发射时：

```text
按逻辑 tensor id 查询 Cache
```

### 命中

- 从 L2 读取；
- 使用 L2 带宽池；
- **命中不会改变 FIFO 顺序**。

### 未命中

- 从 DDR 读取；
- 使用 DDR 带宽池；
- DDR 读取完成后写入 L2；
- 容量不足时按 FIFO 淘汰。

### 特殊情况

若：

```text
单个 Tensor size > L2 容量
```

则该 Tensor 不缓存。

---

## 9.4 结果要求

正文需要：

- 给出 `1~5` 核下：
  - 无 L2；
  - 有只读 L2；

  两种配置的对比曲线。

同核数下 L2 加速比：

```text
speedup_L2
= Makespan_without_L2
  / Makespan_with_L2
```

附录逐用例报告：

- 无 L2 Makespan；
- 有 L2 Makespan；
- 总额外数据搬运量；
- Cache 命中率。

---

# 10. 官方核内调度器行为

参赛算法**不需要直接给出每个 Op 的具体执行时刻**。

官方评估器会对每个 Task 自动完成以下步骤。

---

## 10.1 Step 1：确定性拓扑排序

先将：

```text
Op-Tensor 二部图
```

转为：

```text
Op-Op 依赖图
```

然后生成确定性操作序列。

问题 2、3 中，会在不破坏依赖的情况下，根据 `core_schedules` 调整合并 Task 内的子图顺序。

---

## 10.2 Step 2：缓存容量检查与换入换出

调度器沿固定操作序列维护：

```text
当前驻留于 L1 / UB 的 Tensor
```

若当前操作申请空间导致容量不足：

1. 从可换出的 Tensor 中选取**下一次使用最晚**者；
2. 若 DDR 没有有效副本，先插入 `COPY_OUT`；
3. 在该 Tensor 下次使用前插入 `COPY_IN`；
4. 重复直到容量满足。

失败条件包括：

- 单个 Tensor 超过目标缓存容量；
- 不存在合法可换出 Tensor。

---

## 10.3 Step 3：多 Pipe 排布

同一核心上：

```text
PIPE_M
PIPE_V
PIPE_MTE2
PIPE_MTE3
```

各自串行。

不同 Pipe 可并行。

某个 Op 只有在以下条件满足时才能发射：

- 数据依赖满足；
- 对应 Pipe 空闲；
- 所属 Task 已激活；
- 存储申请顺序允许；
- L1/UB 容量足够。

Tensor 最后一次被使用完成后释放空间。

---

# 11. 官方多核模拟器行为

## 11.1 Op 状态

每个操作依次经历：

```text
pending
  ↓
ready
  ↓
running
  ↓
done
```

---

## 11.2 ready 条件

操作进入 ready 需满足：

- 所属 Task 已激活；
- 全部核内前驱完成；
- 跨核释放时间已到；
- 如需申请缓存，当前轮到其申请；
- L1/UB 容量足够。

---

## 11.3 DDR 带宽竞争

所有访问 DDR 的：

- `COPY_IN`；
- `COPY_OUT`；
- 缓存换入；
- 缓存换出；

共同竞争：

```text
60 bytes/cycle
```

若同一时刻存在多个 DDR 搬运：

```text
公平共享 DDR 总带宽
```

因此单个 COPY 的实际完成时间会随并发搬运数量动态变化。

---

## 11.4 问题 3 的 L2 带宽池

L2 命中的读取使用独立带宽池：

```text
250 bytes/cycle
```

DDR 和 L2：

```text
两个独立带宽池
```

彼此互不占用带宽。

---

# 12. 输入计算图的重要保证

官方测试图保证：

1. 每个 DDR 输入在原始计算图中只搬入一次；
2. 每个最终结果只搬出一次；
3. 中间计算不直接与 DDR 交换数据；
4. 单个计算操作的输入、输出总量不超过相应核内缓存容量。

注意：

第 4 条只保证**单个操作本身可执行**，不保证多个 Tensor 同时驻留时一定不超容量。

---

# 13. 官方文件与脚本

目录主要包括：

```text
README.md
code/
data/
docs/
```

主要评估器：

```text
code/multicore_cut_evaluate_problem_1.py
code/multicore_cut_evaluate_problem_2.py
code/multicore_cut_evaluate_problem_3.py
```

核内调度：

```text
code/schedule_step1.py
code/schedule_step2.py
code/schedule_step3.py
```

示例方案生成：

```text
code/stub_multicore_cut_and_schedule.py
```

`stub` 只保证格式合法，不是性能基线。

---

# 14. 评估器输出

每个问题通常输出三类文件：

```text
<case>_<problem>_res.json
<case>_<problem>_trace.json
<case>_<problem>_log.txt
```

其中：

### `res.json`

主要数值指标，例如：

- Makespan；
- 数据搬运量；
- Cache 指标。

### `trace.json`

Perfetto / Chrome Trace Event JSON。

可用于查看：

- 各核时间线；
- Pipe 使用；
- COPY；
- 并行关系；
- 等待。

### `log.txt`

精简信息：

- 配置；
- Makespan；
- 搬运量；
- 缓存峰值；
- Task / 子图时间段。

---

# 15. 三道题对比总表

| 项目 | 问题 1 | 问题 2 | 问题 3 |
|---|---|---|---|
| 场景 | A | B | B + L2 |
| 一个 Task 包含 | 1 个子图 | 同一核全部子图 | 同一核全部子图 |
| 同核跨子图数据 | 经 DDR | 可留 L1/UB | 可留 L1/UB |
| 跨核数据 | DDR | DDR | DDR 或 L2 命中读取 |
| 同核 Task 切换等待 | 100 cycles | 不适用 | 不适用 |
| 跨核同步延迟 | 1000 cycles 前驱等待 | COPY 后 500 cycles | COPY 后 500 cycles |
| L1 | 512 KiB | 512 KiB | 512 KiB |
| UB | 128 KiB | 128 KiB | 128 KiB |
| DDR 总带宽 | 60 B/cycle | 60 B/cycle | 60 B/cycle |
| L2 | 无 | 无 | 1 MiB，只读 FIFO |
| L2 带宽 | — | — | 250 B/cycle |
| 主指标 | Makespan | Makespan | Makespan |
| 次指标 | 额外搬运 | 额外搬运 | 额外搬运、L2 命中率 |

---

# 16. 后续 Agent 最需要牢记的约束

后续建模或代码 Agent 应默认以下事实，不必再次读取 DOCX 确认：

1. **输出只决定“切图 + 子图分核 + 同核子图顺序”，不直接调度单个 Op。**
2. **必须给每个非 COPY Op 分配且仅分配一个 `sgid`。**
3. **每个 `sgid` 必须在 `core_schedules` 中恰好出现一次。**
4. **子图依赖图必须为 DAG。**
5. **同核子图顺序不得违反依赖。**
6. **问题 1 中即使两个子图在同一核，跨子图数据仍需 DDR 中转。**
7. **问题 2/3 中同一核所有子图属于一个 Task，可直接复用 L1/UB。**
8. **问题 2/3 只有跨核数据边才插入边界 COPY。**
9. **DDR 是全局共享瓶颈，所有 DDR 搬运公平共享 60 B/cycle。**
10. **Cube 与 Vector 可在同一核心并行。**
11. **L1/UB 不足时官方核内调度器会自动插入换出/换入。**
12. **过大的 Task/子图可能因缓存压力而产生额外换入换出。**
13. **问题 3 的 L2 只缓存 COPY_IN 数据，按逻辑 tensor id 查询，替换策略为 FIFO。**
14. **L2 命中不会刷新 FIFO 顺序。**
15. **L2 与 DDR 有独立带宽池。**
16. **最终性能必须以官方 evaluator 为准，不能只用静态计算量估计。**
17. **算法应避免暴力枚举；题面建议单用例约 5~10 分钟内产生结果。**
18. **实验中 1 核基准固定使用官方启发式核内调度结果。**

---

# 17. 一句话理解三问

```text
问题 1：
如何在“切得细可并行，但切图必经 DDR”之间权衡？

问题 2：
允许同核缓存复用后，如何联合优化核心分区、并行度和 L1/UB 压力？

问题 3：
在问题 2 基础上，如何进一步利用共享只读 L2 减少多核重复读取 DDR？
```

---

## 文档来源

本文根据 `challenge.docx` 整理，仅压缩题面信息，不改变题目定义。后续若实现算法、复现实验或解释评估结果，优先以本速查文档作为上下文；只有遇到评估器实现细节争议时再查原始 DOCX / `docs/` / 官方 Python 源码。
