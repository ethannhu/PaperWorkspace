# Q1 消融实验设计指导

## 1. 实验目的

本实验针对问题 1（场景 A）的多核切图与调度算法进行消融分析，验证完整算法中各核心机制对最终性能的贡献。

Q1 的主要优化目标为最小化多核执行 Makespan，同时兼顾额外 DDR 数据搬运量。场景 A 中，每个子图独立构成一个 Task，所有跨子图数据均需要经过 DDR 中转。因此，算法需要在以下因素之间进行权衡：

1. 子图划分粒度；
2. 跨子图通信开销；
3. 多核并行度；
4. 关键路径执行效率；
5. 多核负载均衡。

完整算法记为 `Full`。

本次消融实验仅考察以下四个核心机制：

- Semantic Partition：语义感知分块；
- Adaptive Granularity：图结构自适应分块粒度；
- Critical-path Affinity：关键路径亲和调度；
- Global Rebalance：全局负载再平衡。

实验采用控制变量原则：每个消融版本仅关闭或替换一个机制，其余算法流程、评测程序、配置参数和测试数据均保持不变。

---

## 2. 实验版本

共设置 5 个算法版本：

| ID | Variant | 修改内容 |
|---|---|---|
| A0 | Full | 完整 Q1 算法 |
| A1 | w/o Semantic Partition | 去除语义感知分块 |
| A2 | w/o Adaptive Granularity | 去除图结构自适应分块粒度 |
| A3 | w/o Critical-path Affinity | 去除关键路径亲和调度 |
| A4 | w/o Global Rebalance | 去除最终全局负载再平衡 |

除指定模块外，其他模块必须与 Full 完全一致。

---

# 3. A0：Full

## 3.1 定义

直接使用当前 `q1_algorithm.py` 中的完整算法，不进行任何修改。

整体流程为：

Graph
  ↓
Graph Feature / Pattern Analysis
  ↓
Semantic Partition
  ↓
Pattern-adaptive Partition Strategy
  ↓
Communication-aware Scheduling
  ↓
Critical-path Affinity（适用图类型）
  ↓
Global Rebalance
  ↓
Final Plan

Full 是所有消融实验的统一对照组。

## 3.2 输出

对每个 case、每个核心数 K ∈ {1,2,3,4,5} 保存：

- Makespan
- Speedup
- Added DDR bytes
- Partition count
- 每个 Core 的 partition 数量
- 每个 Core 的估计计算负载
- graph_pattern
- algorithm diagnostics

建议所有实验统一保存为 CSV/JSON，避免后期重新运行。

---

# 4. A1：w/o Semantic Partition

## 4.1 实验目的

验证“利用算子语义和计算图局部结构形成子图”是否优于仅依据 DAG 拓扑关系进行机械分块。

需要回答：

> 在相近分块粒度下，语义感知分块是否能够形成通信关系更合理、调度效率更高的子图？

## 4.2 Full 中的机制

Full 使用 `semantic_partition()` 生成初始 partition。

语义分块应尽可能保留：

- Compute → Elementwise 等局部计算结构；
- 连续计算链；
- Reduce / activation 等语义关系；
- 对 fan-in / fan-out 等复杂结构进行保守处理。

## 4.3 消融方法

禁止使用“一 Op 一个 partition”作为该实验的主要 baseline。

否则同时改变了：

- 是否使用语义；
- partition 数量；
- partition 大小；
- 通信边界数量。

无法单独说明 Semantic Partition 的贡献。

应使用：

### Topology-only Chunking

首先获得 Op DAG 的拓扑序：

    v1, v2, ..., vn

然后按照固定大小连续划分：

    P1 = {v1, ..., vk}
    P2 = {v(k+1), ..., v(2k)}
    ...

划分时只允许使用：

- 拓扑顺序；
- max_ops；
- max_cycles。

禁止使用：

- op 类型；
- semantic role；
- motif；
- Compute / Elementwise / Reduction 类别；
- Tensor 共享语义。

为了公平比较，Topology-only 方法的平均 partition size 应尽可能接近 Full。

## 4.4 保持不变

A1 之后仍然使用 Full 的：

- 图模式识别；
- 自适应粒度参数；
- 调度器；
- Critical-path Affinity；
- Global Rebalance。

即只替换：

    semantic_partition()

为：

    topology_only_partition()

## 4.5 重点指标

重点记录：

- Makespan
- Speedup
- Added DDR bytes
- Partition count
- Boundary count

尤其检查：

    Full 与 A1 的 partition count 是否处于相近量级。

如果 partition 数量差异极大，需要在论文中说明，否则无法排除“分块大小不同”造成的影响。

---

# 5. A2：w/o Adaptive Granularity

## 5.1 实验目的

验证根据计算图结构动态调整 `max_ops` 和 `max_cycles` 是否有必要。

需要回答：

> 不同拓扑结构的计算图是否需要不同的子图粒度？

Full 中不同图类型使用不同的 partition 参数，因此该实验主要验证：

    graph structure
          ↓
    partition granularity
          ↓
    communication / parallelism trade-off

是否真实存在。

## 5.2 Full 中的机制

Full 根据以下信息动态决定 partition 大小：

- graph pattern；
- graph family；
- depth；
- width；
- residual / gated / wide 等结构特征。

形式上可以表示为：

    (max_ops, max_cycles)
        = f(pattern, depth, width)

## 5.3 消融方法

A2 中关闭上述参数自适应。

所有 case 使用完全相同的：

    max_ops = C_ops
    max_cycles = C_cycles

推荐首先使用：

    max_ops = 24
    max_cycles = 32000

如果该参数明显偏向某类图，可以使用 Full 在所有 case 上参数的中位数作为统一值。

一旦确定，所有测试必须使用同一组参数，不允许逐 case 调参。

## 5.4 保持不变

保留：

- Semantic Partition；
- Pattern classification；
- 原有调度器选择；
- Critical-path Affinity；
- Global Rebalance。

Pattern classification 可以继续执行，但不得再用于决定 `max_ops/max_cycles`。

也就是说：

Full:

    parameter = f(pattern, width, depth)

A2:

    parameter = constant

## 5.5 重点指标

除了总体平均性能，必须按照 graph pattern / family 分组统计。

例如：

| Graph Family | Full | Fixed Granularity | ΔMakespan |
|---|---:|---:|---:|
| WIDE | | | |
| COMPLEX | | | |
| MIXED | | | |
| OTHER | | | |

该实验不应只报告所有 case 的总体平均值。

因为 Adaptive Granularity 的理论作用本身就是：

> 对不同结构使用不同粒度。

因此“分图类型结果”比总体平均值更加重要。

---

# 6. A3：w/o Critical-path Affinity

## 6.1 实验目的

验证关键路径感知和关键路径同核亲和机制是否能够降低复杂 DAG 的 Makespan。

需要回答：

> 在 partition 完全相同的情况下，对关键路径进行特殊调度是否能够进一步降低执行时间？

这是一个纯调度消融实验。

## 6.2 Full 中的机制

Full 首先在 partition DAG 上计算 heavy critical path。

路径代价综合考虑：

    partition compute cycles

以及：

    edge communication cost

可以抽象表示为：

    CP = argmax_P [
        Σ C_v +
        Σ C_uv
    ]

随后 Complex Graph 调度过程中：

1. Critical Path partition 获得更高调度优先级；
2. 优先保持关键依赖的 Core affinity；
3. 若破坏 affinity，则加入 split penalty。

## 6.3 消融方法

A3 必须保持 partition 结果完全不变。

仅将：

    _schedule_complex_partitions(...)

替换为：

    _schedule_partitions(...)

即：

Full:

    semantic/adaptive partition
              ↓
    critical-path-aware scheduler

A3:

    identical partitions
              ↓
    normal list scheduler

禁止重新切图。

这样可以确保性能差异完全来自调度策略，而不是 partition 变化。

## 6.4 重点测试对象

重点观察：

- COMPLEX
- Residual
- Attention-like
- 深依赖链图

对于 WIDE 等关键路径很短、并行度很高的图，允许该机制贡献很小甚至没有贡献。

这本身也是合理实验结果。

## 6.5 重点指标

记录：

- Makespan
- Speedup
- Critical Path Length
- Critical Path Cycles
- Critical-path Cross-core Count

定义：

    N_CP_cross =
        关键路径上前后两个 partition
        被分配到不同 Core 的边数

建议额外统计：

    ΔT_CP =
        (T_without_CP - T_full)
        / T_full

按 graph family 分组。

---

# 7. A4：w/o Global Rebalance

## 7.1 实验目的

验证初始启发式调度完成后的全局负载再平衡是否能够进一步降低核间负载不均衡。

需要回答：

> 局部通信/关键路径优化之后，是否仍然需要全局负载修正？

## 7.2 Full 中的机制

Full 在 partition 和初始 core schedule 完成后执行：

    rebalance_core_orders(...)

因此完整流程为：

Initial Schedule
      ↓
Global Rebalance
      ↓
Final Schedule

## 7.3 消融方法

直接跳过：

    rebalance_core_orders()

使用初始 `core_orders` 作为最终结果。

其他所有部分完全不变。

这是四个消融中控制变量最严格的一组。

## 7.4 重点指标

除 Makespan 外必须计算核间负载不均衡程度。

对 Core k 定义估计负载：

    L_k = Σ cycles(P_i)

其中 P_i 为分配到 Core k 的 partition。

推荐使用负载变异系数：

    CV_load = std(L_1,...,L_K)
              -----------------
              mean(L_1,...,L_K)

同时可以记录：

    I_load =
        max(L_k)
        --------
        mean(L_k)

越接近 1 表示越均衡。

需要验证：

    Global Rebalance
          ↓
    lower load imbalance
          ↓
    lower Makespan

---

# 8. 统一实验配置

## 8.1 测试数据

必须使用同一套正式 case。

所有 Variant：

- 不允许删除困难 case；
- 不允许针对某个 Variant 选择不同 case；
- 不允许针对单个 case 手动修改参数。

## 8.2 核心数量

完整实验运行：

    K ∈ {1, 2, 3, 4, 5}

消融分析重点可以使用：

    K = 4

原因是 4 核下已经具有明显的：

- 多核并行；
- 通信竞争；
- 负载均衡；
- 调度空间。

同时仍需要保存 2、3、5 核结果，用于检查结论是否随核心数变化。

## 8.3 官方评测指标

必须使用官方 evaluator 输出：

- Makespan；
- Added Data Movement Bytes。

禁止使用算法内部 estimated makespan 代替官方 Makespan。

---

# 9. Speedup 计算

对于 case i、核心数 K：

    S_i,K = T_i,1 / T_i,K

其中 T_i,1 使用题目规定的单核基准。

平均加速比必须按照：

    S̄_K = (1/N) Σ S_i,K

计算。

禁止使用：

    Σ T_i,1
    --------
    Σ T_i,K

作为平均加速比。

---

# 10. 推荐结果表

## 10.1 主消融表

优先使用 4 核结果：

| Variant | Avg. Speedup ↑ | Avg. Makespan ↓ | Extra DDR ↓ | #Partitions | Load CV ↓ |
|---|---:|---:|---:|---:|---:|
| Full | | | | | |
| w/o Semantic Partition | | | | | |
| w/o Adaptive Granularity | | | | | |
| w/o Critical-path Affinity | | | | | |
| w/o Global Rebalance | | | | | |

不要只记录最终平均值。

必须同时保存 per-case 原始数据。

---

# 11. 相对性能变化

为了使不同 case 可比较，推荐计算相对于 Full 的 Makespan 变化率：

    ΔT =
        T_ablation - T_full
        -------------------
             T_full
        × 100%

其中：

- ΔT > 0：删除该机制后性能下降；
- ΔT ≈ 0：该机制对该 case 影响很小；
- ΔT < 0：删除该机制后反而更快。

注意：

如果出现 ΔT < 0，不得删除该 case 或修改结果。

这通常说明当前 heuristic 在该类型图上存在副作用，也是有价值的分析结果。

---

# 12. 分 Graph Family 分析

建议按照已有 `graph_pattern` / `graph_family` 对 case 分组。

至少考虑：

    WIDE
    COMPLEX
    MIXED
    OTHER

统计：

    mean(ΔT | family)

最终形成：

| Family | w/o Semantic | w/o Adaptive | w/o CP | w/o Rebalance |
|---|---:|---:|---:|---:|
| WIDE | | | | |
| COMPLEX | | | | |
| MIXED | | | | |
| OTHER | | | | |

该表用于回答：

    哪个模块主要解决哪类图？

预期分析方向，而非预设实验结论：

- Semantic Partition：可能具有较普遍作用；
- Adaptive Granularity：不同 family 之间差异应较明显；
- Critical-path Affinity：重点观察 COMPLEX；
- Rebalance：重点观察具有较高并行度但初始负载不均衡的图。

必须以实际实验结果为准，不得为了符合预期修改分类或实验数据。

---

# 13. 推荐绘图

## Figure A：主消融柱状图

横轴：

    Full
    w/o Semantic
    w/o Adaptive
    w/o CP Affinity
    w/o Rebalance

纵轴：

    4-core Average Speedup

用于展示完整算法及各模块删除后的总体性能。

---

## Figure B：Makespan Degradation

横轴为四个消融模块。

纵轴：

    Mean ΔMakespan (%)

即：

    mean(
        (T_ablation - T_full)
        / T_full
    )

相比直接画 Makespan，该指标更容易展示不同模块的相对贡献。

---

## Figure C：Graph-family Ablation Heatmap

行：

    WIDE
    COMPLEX
    MIXED
    OTHER

列：

    Semantic
    Adaptive
    CP Affinity
    Rebalance

单元格：

    mean ΔMakespan (%)

该图用于展示算法机制与计算图结构之间的对应关系。

---

# 14. 实验结果保存格式

建议每次运行保存一行：

    case
    variant
    num_cores
    graph_family
    graph_pattern
    makespan
    speedup
    added_bytes
    partition_count
    boundary_count
    load_cv
    max_load_ratio
    critical_path_length
    critical_path_cycles
    cp_cross_core_edges

例如：

    case_001,full,4,WIDE,...
    case_001,no_semantic,4,WIDE,...
    case_001,no_adaptive,4,WIDE,...
    case_001,no_cp,4,WIDE,...
    case_001,no_rebalance,4,WIDE,...

推荐最终统一输出：

    ablation_results.csv

这样后续所有表格和图片都从同一个 CSV 自动生成。

---

# 15. 实现原则

为了保证实验有效，修改代码时遵循以下原则：

1. 不直接修改 Full 算法逻辑；
2. 使用 `ablation` 参数控制不同 Variant；
3. 每次只关闭一个机制；
4. 所有 Variant 共用相同 graph feature；
5. 所有 Variant 共用相同 evaluator；
6. 不根据测试结果逐 case 调参；
7. 保存 diagnostics，不能只保存最终 Makespan；
8. 保留所有失败或性能退化的 case。

推荐接口：

    build_plan(
        graph,
        num_cores=4,
        ablation=None,
    )

其中：

    None
        -> Full

    "no_semantic"
        -> A1

    "no_adaptive"
        -> A2

    "no_critical_path"
        -> A3

    "no_rebalance"
        -> A4

这样能够最大程度保证各消融版本共享同一套代码，减少由于复制多份算法文件产生的非预期差异。

---

# 16. 最终希望验证的逻辑链

本次消融实验不是简单证明“Full 最好”，而是分别验证四条机制链。

### Semantic Partition

    semantic structure
        ↓
    better partition boundaries
        ↓
    lower communication / better locality
        ↓
    lower Makespan

### Adaptive Granularity

    graph structure
        ↓
    adaptive partition size
        ↓
    balance communication and parallelism
        ↓
    robust performance across graph families

### Critical-path Affinity

    critical path identification
        ↓
    critical-task priority / affinity
        ↓
    fewer expensive critical-path separations
        ↓
    lower critical-path completion time
        ↓
    lower Makespan

### Global Rebalance

    initial heuristic schedule
        ↓
    detect load imbalance
        ↓
    redistribute movable partitions
        ↓
    lower inter-core load imbalance
        ↓
    lower Makespan

最终论文中的消融分析应围绕以上四条因果链展开，而不是仅报告“删除模块后性能下降 X%”。

尤其需要结合：

- partition count；
- Added DDR；
- critical-path cross-core edges；
- load CV；

解释 Makespan 为什么发生变化。

# Q2 消融实验设计指导

## 1. 实验目的

本实验针对问题 2（场景 B）的多核切图与调度算法进行消融分析，验证完整算法中各核心机制对 Makespan 和额外 DDR 数据搬运的贡献。

与 Q1 不同，Q2 中同一核心上的所有子图组成一个 Task：

- 同核子图之间的数据可以保留在 L1/UB 中并直接复用；
- 同核依赖不需要额外 COPY_OUT / COPY_IN；
- 跨核依赖需要经过 DDR；
- 跨核数据传输需要额外承担固定同步延迟；
- 同核数据长期驻留也可能增加 L1/UB 压力。

因此 Q2 的核心优化矛盾为：

    多核并行度
          ↕
    跨核通信开销
          ↕
    同核数据复用
          ↕
    核间负载均衡

本次消融实验重点验证以下四个机制：

1. Semantic & Structure-aware Partition
2. Communication-aware Core Affinity
3. Critical-path Sticky Scheduling
4. Replicated-branch Adaptive Core Control

完整算法记为 `Full`。

实验遵循控制变量原则：

> 每个消融版本只删除或替换一个机制，其余分区、调度、评测配置和测试数据尽可能保持不变。

---

# 2. 实验版本

设置以下 5 个算法版本：

| ID | Variant | 删除/替换机制 | 主要验证问题 |
|---|---|---|---|
| B0 | Full | 无 | 完整 Q2 |
| B1 | w/o Structure-aware Partition | 去除图结构自适应分区策略 | 图结构知识是否改善切图 |
| B2 | w/o Communication Affinity | 去除跨核通信感知 | 同核数据局部性是否重要 |
| B3 | w/o Critical-path Sticky | 去除关键路径亲和调度 | 关键路径保护是否有效 |
| B4 | w/o Adaptive Core Control | 禁用复制分支的有效核数控制 | 是否存在“核越多反而越慢” |

---

# 3. B0：Full

## 3.1 定义

直接使用当前 `q2_algorithm.py` 的完整 Q2 算法。

总体流程：

    Input DAG
        ↓
    Graph Feature Extraction
        ↓
    Graph Pattern Classification
        ↓
    Structure-aware Semantic Partition
        ↓
    Pattern-specific Strategy
        ↓
    Communication-aware Core Assignment
        ↓
    Critical-path / Component-aware Scheduling
        ↓
    Global Rebalance
        ↓
    Final Schedule

Full 作为全部消融版本的统一对照。

---

## 3.2 Full 中需要保留的诊断信息

对每个 case、每个核心数 K ∈ {1,2,3,4,5} 保存：

- graph_family
- graph_pattern
- partition_count
- max_ops
- max_cycles
- scheduler
- critical_path_length
- critical_path_cycles
- effective_core_count
- global_rebalance
- 每核 partition 数量
- 每核估计计算负载

官方 evaluator 结果保存：

- Makespan
- Added Data Movement Bytes
- Speedup

---

# 4. B1：w/o Structure-aware Partition

## 4.1 实验目的

验证根据计算图结构选择不同分块策略是否能够改善 Q2 性能。

需要回答：

> 对所有计算图使用统一语义分区策略，与根据 WIDE、COMPLEX、MIXED 等图结构选择不同分区策略相比，性能有何变化？

该实验重点验证：

    Graph Pattern
         ↓
    Partition Strategy
         ↓
    Parallelism / Locality
         ↓
    Makespan

---

## 4.2 Full 中的机制

Full 首先进行 graph pattern classification，然后根据 family 选择：

    WIDE
        → _wide_plan()

    COMPLEX
        → _complex_plan()

    MIXED
        → _mixed_plan()

    OTHER
        → _semantic_plan()

不同策略使用不同的：

- max_ops
- max_cycles
- singleton repair
- topological coalescing
- scheduler

例如 WIDE 图会进一步进行 bounded topological coalescing，以减少过细的 semantic partition。

COMPLEX 图则允许更大的语义块，以避免复杂 residual / normalization chain 被过度切分。

---

## 4.3 消融方法

B1 中关闭 graph-family-specific partition strategy。

所有 case 统一采用：

    semantic_partition(features)

作为 partition。

即：

Full:

    classify graph
         ↓
    WIDE / COMPLEX / MIXED / OTHER
         ↓
    specialized partition

B1:

    all graphs
         ↓
    default semantic_partition

注意：

为了使该实验主要测试“分区策略”，分区完成后仍然允许使用 Full 对应的调度器。

也就是说：

> 尽量只改变 partitions，不同时删除通信感知或关键路径调度。

---

## 4.4 禁止的实现

不能使用：

    one operator = one partition

作为 B1。

这会导致 partition 数量发生数量级变化，使实验同时改变：

- partition granularity；
- communication boundaries；
- cache pressure；
- scheduler search space。

无法公平评价结构感知分区的贡献。

---

## 4.5 重点指标

记录：

    Makespan
    Speedup
    Added DDR
    Partition Count
    Cross-core Edge Count
    Cross-core Bytes

定义：

    B_cross =
        Σ size(e)

其中 e 为跨 Core 的 partition dependency。

重点比较：

    Full vs B1

在不同 graph family 下的差异。

---

# 5. B2：w/o Communication Affinity

## 5.1 实验目的

这是 Q2 最重要的消融实验之一。

验证：

> 在场景 B 中，显式考虑跨核通信和同核数据复用，是否能够改善多核调度性能？

Q2 中，同核 dependency 不需要 DDR 搬运，而跨核 dependency 需要：

    COPY_OUT
       ↓
      DDR
       ↓
    synchronization
       ↓
    COPY_IN

因此 Core assignment 本身直接决定通信量。

---

## 5.2 Full 中的机制

Full 调度一个 partition p 到 Core k 时，会估计：

    dependency_ready

    candidate_edge_bytes

    core_time

    pipe_load

    DDR lower bound

对于已经放置的前驱：

如果：

    core(pred) == k

则数据可以保持同核。

如果：

    core(pred) != k

则加入跨核通信：

    transfer_delay =
        500
        + ceil(edge_bytes / 60)

因此 Full 实际在优化：

    compute parallelism
           +
    communication locality

而不是单纯进行负载均衡。

---

# 6. B2 消融方法

保持：

- partitions 完全相同；
- ready-task priority 完全相同；
- critical rank 完全相同。

仅在选择目标 Core 时删除 communication information。

消融后的 Core 选择只考虑：

    earliest compute finish
    +
    pipe balance

不考虑：

    edge_bytes
    cross-core transfer
    DDR lower bound
    500-cycle synchronization penalty

可以抽象为：

Full:

    Score(p,k) =
        f(
          finish_time,
          dependency_delay,
          DDR_bytes,
          pipe_load
        )

B2:

    Score(p,k) =
        f(
          finish_time,
          pipe_load
        )

---

## 6.1 注意

不要使用 Random Core Assignment。

Random baseline 太弱，不能说明 communication-aware scheduling 的真实贡献。

B2 应该仍然是一个合理的：

    load-aware list scheduler

只是它：

> 不知道通信代价。

这样实验才能回答：

    Communication awareness
        是否真正有用？

---

# 7. B2 重点指标

除了 Makespan，必须记录：

    Cross-core Edge Count

    Cross-core Bytes

    Added DDR Bytes

    Load CV

定义跨核通信字节：

    B_cross =
        Σ size(e)
      e ∈ E_cross

建议进一步计算：

    Cross-core Ratio =
        B_cross
        -------
        B_dependency

如果：

    B2:
        Load balance ≈ Full

但：

    Cross-core Bytes ↑
    Added DDR ↑
    Makespan ↑

则可以比较有力地证明：

> Q2 的收益不是单纯来自负载均衡，而来自通信感知的 Core affinity。

---

# 8. B3：w/o Critical-path Sticky

## 8.1 实验目的

验证对复杂 DAG 中关键依赖链进行优先调度和同核保护是否能够降低 Makespan。

需要回答：

> 在 partition 完全相同的情况下，关键路径感知是否能够改善最终执行时间？

---

## 8.2 Full 中的机制

对于 COMPLEX / MIXED 等图，Full 首先计算 partition DAG 上的 heavy critical path：

    CP = argmax_P [
        Σ compute_cycles
        +
        Σ communication_cost
    ]

随后调度过程中：

1. Critical Path partition 优先进入调度；
2. 优先保持关键前驱所在 Core；
3. 将重要 predecessor 的 Core 作为 preferred core；
4. 若将 partition 分配到其他 Core，则加入 split penalty。

MIXED 图使用较弱的 soft sticky penalty。

---

# 9. B3 消融方法

必须保证：

    partitions_B3 == partitions_Full

只替换 scheduler。

Full:

    _schedule_complex_partitions(...)

替换为：

    _schedule_partitions(...)

即：

    Same Partitions
          ↓
    Normal Communication-aware
    List Scheduling

普通 scheduler 仍然：

- 感知跨核通信；
- 感知 DDR；
- 感知 Pipe load。

唯一删除：

    critical-path priority
    +
    sticky affinity penalty

因此这是一个非常严格的控制变量实验。

---

# 10. B3 重点指标

记录：

    Makespan
    Speedup
    Critical Path Length
    Critical Path Cycles
    CP Cross-core Edge Count
    CP Cross-core Bytes

定义：

    N_CP_cross =
        Σ 1[
            core(P_i)
            !=
            core(P_i+1)
          ]

其中：

    P_i, P_i+1 ∈ Critical Path

以及：

    B_CP_cross =
        Σ edge_size(P_i, P_i+1)

重点分析：

    COMPLEX
    MIXED

对于高度并行的 WIDE 图，Critical-path Sticky 贡献很小属于正常现象。

---

# 11. B4：w/o Adaptive Core Control

## 11.1 实验目的

这是 Q2 中非常有特色的一组消融。

验证：

> 对通信占主导的短复制分支，是否应该主动限制参与计算的核心数量？

通常：

    more cores
        → more parallelism

但在 Q2 中也可能：

    more cores
        ↓
    more branch splitting
        ↓
    more cross-core COPY
        ↓
    more DDR traffic
        ↓
    synchronization overhead
        ↓
    worse Makespan

因此：

    最优有效核心数
        不一定等于
    物理可用核心数

---

## 11.2 Full 中的机制

对于 replicated narrow CNN 等特定结构，Full 首先识别：

    narrow
    +
    deep
    +
    replicated branches

然后根据 critical cycles 判断是否限制有效核心数量。

对于较短的复制链：

    effective_cores =
        min(num_cores, 2)

否则：

    effective_cores =
        num_cores

同时，如果进行了 Core Cap，后续 Global Rebalance 不允许重新把 partition 移动到被主动闲置的 Core。

---

# 12. B4 消融方法

删除：

    effective_core_count

限制。

无论 graph pattern 如何，始终使用：

    effective_cores = num_cores

即：

Full:

    if communication-dominated short replicas:
        use fewer cores

B4:

    always use all available cores

其他逻辑全部保持不变：

- partitions 相同；
- component scheduling 相同；
- branch packing 相同；
- core selection 规则相同。

只取消：

    adaptive core-count control

---

# 13. B4 重点指标

重点测试：

    K = 2,3,4,5

而不是只看 4 核。

因为该实验研究的是：

    Core Count
        ↕
    Communication Cost

建议对相关 case 画：

    x-axis:
        number of physical cores

    y-axis:
        Makespan

两条曲线：

    Full Adaptive Core Control

    Always Use All Cores

同时记录：

    effective_core_count
    cross-core bytes
    added DDR bytes
    core utilization

---

# 14. 统一实验配置

所有版本使用：

    same cases
    same config.txt
    same evaluator
    same graph features
    same random state（若存在）
    same machine/environment

禁止：

- 删除性能下降 case；
- 针对不同 variant 使用不同测试集；
- 针对单个 case 人工调参；
- 修改官方 evaluator；
- 使用算法内部估计 Makespan 代替官方 Makespan。

---

# 15. 核数设置

全部实验运行：

    K ∈ {1,2,3,4,5}

其中：

    K = 4

作为主消融表。

但 B4：

    Adaptive Core Control

必须重点展示：

    K = 2,3,4,5

因为其研究对象本身就是有效核心数量。

---

# 16. Speedup

对 case i：

    S_i,K =
        T_i,1
        -----
        T_i,K

平均加速比：

             N
    S̄_K = 1/N Σ S_i,K
            i=1

必须：

    先计算每个 case 的 Speedup
    再对所有 case 求平均。

---

# 17. 推荐诊断指标

每个实验至少保存：

    case
    variant
    num_cores

    graph_family
    graph_pattern

    makespan
    speedup
    added_ddr_bytes

    partition_count

    cross_core_edges
    cross_core_bytes

    load_cv

    critical_path_length
    critical_path_cycles

    cp_cross_core_edges
    cp_cross_core_bytes

    effective_core_count

---

# 18. Load Balance

定义 Core k 的计算负载：

    L_k =
        Σ cycles(P)

其中：

    P assigned to Core k

定义：

                 std(L)
    CV_load = ---------------
                mean(L)

另外记录：

                 max(L)
    MaxRatio = ---------------
                mean(L)

这些指标用于区分：

    communication optimization

和：

    load balancing optimization

的贡献。

---

# 19. 相对 Makespan 退化

对每个 case 和每个消融：

                    T_ablation - T_full
    ΔT (%) = 100 × ---------------------
                          T_full

解释：

    ΔT > 0
        删除模块后性能下降

    ΔT ≈ 0
        模块对该 case 影响较小

    ΔT < 0
        Full heuristic 在该 case 上产生负收益

必须保留 ΔT < 0 的 case。

不能为了得到“所有模块均有效”的结论而删除异常结果。

---

# 20. 按 Graph Family 分组

建议统计：

    mean(ΔT | graph_family)

形成：

| Graph Family | w/o Structure | w/o Comm. | w/o CP | w/o Core Control |
|---|---:|---:|---:|---:|
| WIDE | | | | |
| COMPLEX | | | | |
| MIXED | | | | |
| OTHER | | | | |

该表用于分析：

    不同算法模块
          ↓
    分别解决什么图结构问题

不要预设每个模块都应该在所有 family 上获得正收益。

---

# 21. 推荐主消融表

使用 4 核结果：

| Variant | Avg. Speedup ↑ | ΔMakespan ↓ | Added DDR ↓ | Cross-core Bytes ↓ | Load CV ↓ |
|---|---:|---:|---:|---:|---:|
| Full | | 0 | | | |
| w/o Structure-aware Partition | | | | | |
| w/o Communication Affinity | | | | | |
| w/o Critical-path Sticky | | | | | |
| w/o Adaptive Core Control | | | | | |

注意：

B4 不能仅根据该表得出结论。

还必须结合 2～5 核结果。

---

# 22. 推荐图表

## Figure A：Q2 主消融柱状图

横轴：

    Full
    w/o Structure
    w/o Comm.
    w/o CP
    w/o Core Control

纵轴：

    4-core Average Speedup

用于展示整体性能。

---

## Figure B：Makespan Degradation

横轴：

    四个消融模块

纵轴：

    Mean ΔMakespan (%)

定义：

    ΔT =
        (T_ablation - T_full)
        / T_full
        × 100%

用于直观展示每个模块被删除后的性能变化。

---

## Figure C：通信开销消融

重点比较：

    Full
        vs
    w/o Communication Affinity

横轴：

    Graph Family

纵轴：

    Cross-core Bytes

或：

    Added DDR Bytes

用于验证：

    Communication Affinity
          ↓
    Cross-core traffic reduction
          ↓
    Makespan reduction

---

## Figure D：有效核心数量实验

重点比较：

    Full
        vs
    w/o Adaptive Core Control

横轴：

    2
    3
    4
    5 cores

纵轴：

    Makespan

挑选存在 replicated narrow branches 的代表性 case。

该图用于展示：

    更多物理核心
        ≠
    更高有效并行度

以及：

    Compute Parallelism
          ↕
    Cross-core Communication

之间的权衡。

---

# 23. 推荐结果 CSV

统一保存：

    q2_ablation_results.csv

字段：

    case
    variant
    num_cores

    graph_family
    graph_pattern

    makespan
    speedup

    added_ddr_bytes

    partition_count

    cross_core_edges
    cross_core_bytes

    load_cv
    max_load_ratio

    critical_path_length
    critical_path_cycles

    cp_cross_core_edges
    cp_cross_core_bytes

    effective_core_count

例如：

    case_001,full,4,...
    case_001,no_structure,4,...
    case_001,no_communication,4,...
    case_001,no_critical_path,4,...
    case_001,no_core_control,4,...

所有论文表格和图片应从该 CSV 自动生成。

---

# 24. 推荐代码接口

不要复制 5 份 q2_algorithm.py。

建议增加统一参数：

    build_plan(
        graph,
        num_cores=4,
        ablation=None,
    )

定义：

    None
        → Full

    "no_structure"
        → B1

    "no_communication"
        → B2

    "no_critical_path"
        → B3

    "no_core_control"
        → B4

每个消融只在对应位置产生最小代码分支。

这样可以避免不同版本代码逐渐产生额外差异。

---

# 25. 四个实验最终验证的因果链

## B1 Structure-aware Partition

    Graph Structure
          ↓
    Specialized Partition
          ↓
    Appropriate Granularity
          ↓
    Locality + Parallelism
          ↓
    Lower Makespan


## B2 Communication Affinity

    Tensor Dependency
          ↓
    Communication-aware Placement
          ↓
    More Same-core Dependencies
          ↓
    Fewer Cross-core Copies
          ↓
    Lower DDR Traffic
          ↓
    Lower Makespan


## B3 Critical-path Sticky

    Partition DAG
          ↓
    Heavy Critical Path
          ↓
    Priority + Core Affinity
          ↓
    Fewer Critical-path Splits
          ↓
    Shorter Critical Completion Time
          ↓
    Lower Makespan


## B4 Adaptive Core Control

    Replicated Branch Structure
          ↓
    Compute / Communication Analysis
          ↓
    Effective Core Count Selection
          ↓
    Avoid Excessive Branch Splitting
          ↓
    Lower Cross-core Traffic
          ↓
    Lower Makespan


# 26. 最终分析原则

消融实验的目标不是证明：

    Full 永远优于所有 Ablation

而是回答：

    为什么 Full 在某些图上更好？

因此论文分析不能只写：

    “删除模块 X 后平均性能下降 Y%。”

应该进一步利用诊断指标解释：

    Structure-aware Partition
        → partition_count / boundary 改变

    Communication Affinity
        → cross-core bytes 改变

    Critical-path Sticky
        → CP cross-core edges 改变

    Adaptive Core Control
        → effective cores / DDR traffic 改变

最终形成：

    算法机制
       ↓
    中间调度特征变化
       ↓
    硬件资源开销变化
       ↓
    Makespan 变化

的完整实验论证链。

# Q3 消融实验设计指导

## 1. 实验目的

本实验针对问题 3 的只读共享 L2 缓存场景进行消融分析。

Q3 在 Q2 场景 B 的基础上增加所有 Core 共享的只读 L2 Cache。L2 仅缓存 COPY_IN 数据，并采用 FIFO 淘汰策略。多个 Core 对相同数据执行 COPY_IN 时，如果数据已经存在于 L2，则可以直接从 L2 读取，而无需再次访问 DDR。

因此 Q3 相比 Q2 的新增优化机会主要来自：

    Shared Input Tensor
            ↓
    Multiple COPY_IN Requests
            ↓
    L2 Cache Reuse
            ↓
    Fewer DDR Reads
            ↓
    Lower Data-transfer Time
            ↓
    Lower Makespan

但 L2 容量有限，并采用 FIFO 淘汰，因此：

    Potential Reuse
        ≠
    Actual Cache Hit

调度顺序会影响共享 Tensor 两次访问之间的距离，从而进一步影响 L2 命中率。

本次设置 4 个核心消融实验：

1. w/o L2 Reuse
2. w/o Cache-aware Priority
3. Global Cache Priority
4. w/o Communication Affinity

完整算法记为 `Full`。

实验重点回答：

- L2 本身能够带来多少收益？
- 显式识别共享 Tensor 是否有必要？
- 为什么缓存优先级只应该在特定图结构中启用？
- Cache locality 与 Core locality 是否需要联合优化？

---

# 2. 实验版本

| ID | Variant | 修改内容 | 验证目标 |
|---|---|---|---|
| C0 | Full | 完整 Q3 | 完整方法 |
| C1 | w/o L2 Reuse | 禁止利用 L2 命中 | 验证 L2 硬件机制的基础收益 |
| C2 | w/o Cache-aware Priority | 删除 cache reuse ready priority | 验证复用感知调度的贡献 |
| C3 | Global Cache Priority | 所有图均强制使用 cache priority | 验证选择性启用策略的必要性 |
| C4 | w/o Communication Affinity | 删除 Core placement 中的跨核通信代价 | 验证 L2 locality 与 Core locality 的互补性 |

其中：

    C0 vs C1

主要回答“L2 是否有价值”。

    C0 vs C2

主要回答“知道 L2 存在以后，是否还需要主动调度”。

    C0 vs C3

主要回答“Cache-aware 是否应该无条件启用”。

    C0 vs C4

主要回答“有 L2 后是否还需要减少跨核通信”。

---

# 3. C0：Full

## 3.1 完整算法

使用当前 Q3 完整算法。

整体流程：

    Graph Feature Extraction
              ↓
    Graph Pattern Classification
              ↓
    Structure-aware Semantic Partition
              ↓
    Cache Reuse Analysis
              ↓
    Communication-aware Scheduling
              +
    Cache-aware Ready Priority
              +
    Critical-path Affinity
              ↓
    Global Rebalance
              ↓
    Q3 L2 Cache Model
              ↓
    Final Schedule

Q3 的基础分区策略继续复用：

- WIDE-specific partition；
- COMPLEX semantic partition；
- MIXED adaptive partition；
- critical-path scheduling；
- replicated component scheduling。

Q3 特有机制主要加入调度优先级。

---

# 4. Q3 Cache Reuse Value

当前算法首先分析一个 Tensor 是否被多个 partition 使用。

对于 Tensor t：

    U(t) = 使用 Tensor t 的 partition 集合

如果：

    |U(t)| > 1

则该 Tensor 存在潜在跨 partition 复用。

定义 partition p 的 Cache Reuse Value：

                ┌
    R(p) =       │ size(t) × (|U(t)| - 1)
                └
              t ∈ Input(p)
              |U(t)| > 1

直观含义：

一个 Tensor：

- 越大；
- 被越多 partition 使用；

其潜在 L2 复用价值越高。

因此：

    Large Shared Tensor
            ↓
    High Cache Reuse Value
            ↓
    Higher Scheduling Priority

---

# 5. Cache-aware Ready Queue

普通 ready queue：

    Priority(p) =
        rank_u(p)

Q3 对满足条件的图加入：

    Priority(p) =
        (
          rank_u(p),
          cache_reuse(p)
        )

即在满足 DAG 依赖的 ready partition 中：

    Criticality
        +
    Cache Reuse Potential

共同影响执行顺序。

注意：

Cache-aware Priority 不是对所有图开启。

当前 Full 仅针对具有明显共享/复制结构的图启用，例如：

    CNN_RESIDUAL

    GATED_SIGMOID_MLP

这是后续 C3 消融的核心。

---

# 6. C1：w/o L2 Reuse

## 6.1 实验目的

验证问题 3 新增 L2 Cache 机制本身能够带来多少性能收益。

需要回答：

> 在 partition 和 Core schedule 完全相同的情况下，如果 COPY_IN 不能利用 L2 命中，性能会发生什么变化？

这是 Q3 最基础的硬件机制消融。

---

## 6.2 控制变量

必须保证：

    partitions_C1
        ==
    partitions_Full

以及：

    core_schedule_C1
        ==
    core_schedule_Full

也就是说：

    不重新切图
    不重新调度

只改变：

    COPY_IN 的数据来源模型。

---

## 6.3 Full

对于 COPY_IN(t)：

    if t in L2:

        L2 → Core

    else:

        DDR → L2 → Core

---

## 6.4 C1

所有 COPY_IN 均视为：

    DDR → Core

或者在 evaluator 支持的情况下：

    强制 L2 miss

即：

    L2 hit rate = 0

---

## 6.5 重点指标

记录：

    Makespan

    DDR Read Bytes

    L2 Read Bytes

    L2 Hit Count

    L2 Miss Count

    L2 Hit Rate

定义：

                    N_hit
    HitRate = ----------------
               N_hit + N_miss

以及字节级命中率：

                     B_hit
    ByteHitRate = ----------------
                  B_hit + B_miss

对于 Tensor size 差异较大的计算图：

> Byte Hit Rate 比简单 Hit Rate 更重要。

---

# 7. C1 希望验证的因果链

    Shared L2
       ↓
    COPY_IN Cache Hit
       ↓
    Lower DDR Read Traffic
       ↓
    Lower DDR Transfer Time
       ↓
    Lower Makespan

因此不要只报告：

    Makespan ↓

还必须验证：

    DDR Read Bytes ↓

否则无法证明性能改善来自 L2。

---

# 8. C2：w/o Cache-aware Priority

## 8.1 实验目的

这是 Q3 最核心的软件机制消融。

验证：

> 在硬件已经提供 L2 的情况下，显式根据 Tensor 复用价值调整 partition 执行顺序是否仍然有必要？

---

## 8.2 Full

Full ready queue：

    ready.sort(
        rank_u,
        cache_reuse,
        topology
    )

对于启用 cache reuse priority 的图：

    cache_reuse(p)

参与 ready partition 的排序。

---

## 8.3 C2

保留：

- L2 Cache；
- FIFO；
- 相同 Cache Size；
- 相同 partition；
- 相同 Core placement objective；
- 相同 critical-path scheduler。

仅删除：

    cache_reuse priority

即：

Full:

    Priority =
        Criticality
        +
        Cache Reuse

C2:

    Priority =
        Criticality Only

---

# 9. C2 重点指标

记录：

    Makespan

    L2 Hit Rate

    L2 Byte Hit Rate

    DDR Read Bytes

    Average Reuse Distance

    Cache Eviction Count

其中推荐额外定义：

## Reuse Distance

对于同一个 Tensor t 的连续两次 COPY_IN：

    d(t) =
        两次访问之间进入 L2 的其他数据总量

也可以简化为：

    两次访问之间的 COPY_IN 数量

更推荐使用：

    Byte Reuse Distance

因为 FIFO Cache 是否淘汰主要取决于中间进入缓存的数据规模。

---

# 10. C2 希望验证的因果链

    Cache-aware Priority
            ↓
    Shared Tensor Consumers
    Scheduled Closer Together
            ↓
    Lower Reuse Distance
            ↓
    Higher L2 Hit Rate
            ↓
    Lower DDR Traffic
            ↓
    Lower Makespan

这是 Q3 最重要的一条实验论证链。

---

# 11. C3：Global Cache Priority

## 11.1 实验目的

Full 并没有在所有 graph pattern 上启用 cache reuse priority。

该实验验证：

> 为什么 Cache-aware Scheduling 不应该无条件应用于所有计算图？

这是一个非常重要的“反向消融”。

传统消融通常是：

    Full
      -
    mechanism

而该实验是：

    Full
      +
    excessive mechanism

用于验证 Full 中：

    selective cache optimization

设计是否合理。

---

# 12. C3 消融方法

Full：

    if pattern in {
        CNN_RESIDUAL,
        GATED_SIGMOID_MLP
    }:

        enable cache priority

    else:

        disable cache priority

C3：

    所有 graph pattern：

        enable cache priority

即：

    _q3_reuse_priority(features)

始终返回：

    True

---

# 13. 为什么该实验有意义

Cache Reuse Priority 并非没有代价。

如果过度强调缓存复用，可能破坏：

    Critical Path Priority

或者：

    Load Balance

或者：

    Natural Topological Locality

例如某个高复用 partition：

    cache_reuse = very high

但它并不是：

    critical partition

若过度提前执行它，可能延迟真正控制 Makespan 的任务。

因此 Q3 实际存在：

        Criticality
             ↕
        Cache Locality
             ↕
        Parallelism

三者之间的权衡。

---

# 14. C3 重点指标

按照 Graph Family / Pattern 分组统计：

    Makespan

    Hit Rate

    Byte Hit Rate

    DDR Read Bytes

    Critical Path Completion Time

尤其关注：

    WIDE
    COMPLEX
    MIXED
    OTHER

如果出现：

    Global Cache Priority
        → Hit Rate ↑
        → Makespan ↑

这是非常有价值的结果。

它说明：

> 最大化 Cache Hit Rate 并不等价于最小化 Makespan。

---

# 15. C4：w/o Communication Affinity

## 15.1 实验目的

验证加入共享 L2 后，是否仍然有必要进行通信感知的 Core placement。

需要回答：

> L2 能减少 DDR 重复读取以后，Core locality 是否仍然重要？

Q3 仍然继承 Q2 的基本特点：

    same-core dependency
        → local reuse

    cross-core dependency
        → communication / synchronization

因此 L2 并不能完全替代：

    Core Affinity

---

# 16. C4 消融方法

保持：

    partitions

    cache reuse priority

    critical path priority

    L2 model

全部不变。

仅从目标 Core 选择函数中删除：

    cross-core edge bytes

    communication delay

    DDR lower bound

Core assignment 仅依据：

    compute finish time

    pipe balance

进行。

禁止使用：

    Random Core Assignment

因为 Random baseline 过弱。

C4 应仍然是一个合理的：

    load-aware scheduler

只是：

> 不感知跨核通信。

---

# 17. C4 重点指标

记录：

    Makespan

    Cross-core Edge Count

    Cross-core Bytes

    L2 Hit Rate

    DDR Read Bytes

    Load CV

用于分析：

    Cache Locality

和：

    Core Locality

之间的关系。

---

# 18. C4 希望验证的因果链

Full：

    Cache-aware Ordering
            +
    Communication-aware Placement
            ↓
    L2 Locality
            +
    Core Locality
            ↓
    Lower Memory Traffic
            ↓
    Lower Makespan

C4：

    Cache-aware Ordering
            +
    Load-only Placement
            ↓
    More Cross-core Dependencies
            ↓
    Communication Cost ↑
            ↓
    Makespan ↑

如果实验支持该现象，则可以说明：

> Q3 不是单独优化 L2，而是在多层存储结构下联合优化数据局部性。

---

# 19. 统一实验配置

所有 Variant 必须使用：

    Same Cases

    Same Core Count

    Same Cache Size

    Same FIFO Policy

    Same DDR Bandwidth

    Same L2 Bandwidth

    Same Graph Features

    Same Evaluator

禁止：

- 删除性能下降 case；
- 对单个 case 人工修改 Cache Priority；
- 对不同 variant 使用不同 Cache Size；
- 修改 evaluator；
- 使用算法内部 estimated Makespan 代替官方 Makespan。

---

# 20. 核心数量

运行：

    K ∈ {1,2,3,4,5}

主消融表推荐：

    K = 4

但 Q3 建议额外分析：

    K
      vs
    Cache Hit Rate

因为随着 Core 数增加：

    Shared Tensor Consumers ↑

可能提高 L2 复用机会；

同时：

    Concurrent Working Set ↑

也可能提高 FIFO 淘汰压力。

因此：

    Core Count
        ↕
    Cache Reuse
        ↕
    Cache Pressure

本身也是值得观察的关系。

---

# 21. 推荐新增 Q3 指标

Q3 除官方指标外，建议重点记录以下缓存指标。

## 21.1 Cache Hit Rate

                    N_hit
    HR = -------------------------
           N_hit + N_miss

---

## 21.2 Byte Hit Rate

                    B_hit
    BHR = ------------------------
           B_hit + B_miss

Byte Hit Rate 应作为主要缓存指标。

---

## 21.3 DDR Read Reduction

相对于无 L2：

                      B_noL2 - B_Q3
    DDR Saving = ----------------------
                          B_noL2

---

## 21.4 Cache Evictions

记录：

    eviction_count

用于分析 FIFO Cache Pressure。

---

## 21.5 Reuse Distance

对于重复 COPY_IN 的 Tensor：

    reuse_distance(t)

统计：

    mean
    median
    P90

推荐优先统计：

    Byte Reuse Distance

而不是单纯访问次数距离。

---

# 22. 相对性能变化

对消融版本 a：

                        T_a - T_full
    ΔT_a (%) = 100 × ----------------
                           T_full

其中：

    ΔT > 0
        → 删除/改变机制后变慢

    ΔT ≈ 0
        → 机制影响较小

    ΔT < 0
        → 当前 Full heuristic
          在该 case 上存在负收益

所有负收益 case 必须保留。

---

# 23. Graph Pattern 分组分析

建议至少按照：

    WIDE
    COMPLEX
    MIXED
    OTHER

统计：

    mean(ΔT | family)

同时针对：

    CNN_RESIDUAL

    GATED_SIGMOID_MLP

单独报告。

因为当前 Cache-aware Priority 主要针对这些结构。

最终形成：

| Graph Family | w/o L2 | w/o Cache Priority | Global Cache Priority | w/o Comm. Affinity |
|---|---:|---:|---:|---:|
| WIDE | | | | |
| COMPLEX | | | | |
| MIXED | | | | |
| OTHER | | | | |

---

# 24. 推荐主消融表

使用 4 Core：

| Variant | Avg. Speedup ↑ | ΔMakespan ↓ | DDR Read ↓ | Byte Hit Rate ↑ | Cross-core Bytes ↓ |
|---|---:|---:|---:|---:|---:|
| Full | | 0 | | | |
| w/o L2 Reuse | | | | 0 | |
| w/o Cache Priority | | | | | |
| Global Cache Priority | | | | | |
| w/o Communication Affinity | | | | | |

注意：

不要只根据 Hit Rate 判断算法优劣。

最终目标仍然是：

    Makespan

Hit Rate 只是解释性能变化的中间指标。

---

# 25. 推荐 Figure A：Q3 主消融

柱状图。

横轴：

    Full
    w/o L2
    w/o Cache Priority
    Global Cache Priority
    w/o Comm. Affinity

纵轴：

    4-core Average Speedup

用于展示整体性能。

---

# 26. 推荐 Figure B：缓存机制因果图

重点比较：

    Full
        vs
    w/o Cache Priority

绘制两组指标：

    Byte Hit Rate

    DDR Read Bytes

用于证明：

    Cache Priority
        ↓
    Hit Rate
        ↓
    DDR Traffic

这一中间机制。

---

# 27. 推荐 Figure C：Hit Rate 与 Makespan

每个 case 一个点。

横轴：

    Δ Byte Hit Rate

纵轴：

    Δ Makespan

比较：

    Full
        vs
    w/o Cache Priority

用于观察：

> Cache Hit 改善是否真正转化为执行时间改善。

该图尤其适合解释异常 case。

---

# 28. 推荐 Figure D：选择性 Cache Priority

按 graph family 绘制：

    Global Cache Priority
        相对于
    Full

的：

    ΔMakespan

重点展示：

    对某些图：
        Cache Priority 有收益

    对另一些图：
        Cache Priority 无收益
        或产生负收益

用于支撑：

    Pattern-aware Cache Optimization

而不是：

    Always Cache-first Scheduling

---

# 29. 推荐 Figure E：Core Count × Cache

横轴：

    1
    2
    3
    4
    5 Core

左纵轴：

    Speedup

右纵轴：

    Byte Hit Rate

选择具有明显共享 Tensor 的代表 case。

用于展示：

    Parallelism
        ↕
    Shared Cache Reuse
        ↕
    Cache Pressure

之间的关系。

---

# 30. 实验结果保存格式

统一保存：

    q3_ablation_results.csv

字段建议：

    case
    variant
    num_cores

    graph_family
    graph_pattern

    makespan
    speedup

    partition_count

    added_ddr_bytes
    ddr_read_bytes

    l2_read_bytes

    cache_hits
    cache_misses
    cache_hit_rate
    cache_byte_hit_rate
    cache_evictions

    mean_reuse_distance
    median_reuse_distance
    p90_reuse_distance

    cross_core_edges
    cross_core_bytes

    load_cv

    critical_path_cycles
    cp_cross_core_edges

例如：

    case_001,full,4,...
    case_001,no_l2,4,...
    case_001,no_cache_priority,4,...
    case_001,global_cache_priority,4,...
    case_001,no_communication,4,...

---

# 31. 推荐代码接口

不要复制多份 q3_algorithm.py。

建议：

    build_plan(
        graph,
        num_cores=4,
        ablation=None,
    )

定义：

    None
        → Full

    "no_l2"
        → C1

    "no_cache_priority"
        → C2

    "global_cache_priority"
        → C3

    "no_communication"
        → C4

其中需要特别注意：

## no_l2

最好由 evaluator / cache simulator 控制。

因为它不应该改变：

    partition

和：

    schedule

否则无法形成严格的硬件消融。

---

# 32. 最终需要验证的四条因果链

## C1：L2 Hardware Reuse

    Shared L2
       ↓
    Cache Hit
       ↓
    DDR Read Reduction
       ↓
    Lower Makespan


## C2：Cache-aware Scheduling

    Shared Tensor Detection
           ↓
    Cache Reuse Priority
           ↓
    Shorter Reuse Distance
           ↓
    Higher Byte Hit Rate
           ↓
    Lower DDR Traffic
           ↓
    Lower Makespan


## C3：Selective Cache Optimization

    Graph Pattern
        ↓
    Identify Reuse-friendly Motif
        ↓
    Selectively Enable Cache Priority
        ↓
    Avoid Criticality / Locality Conflict
        ↓
    Better Overall Makespan


## C4：Hierarchical Locality

    Cache-aware Ordering
           +
    Communication-aware Placement
           ↓
    L2 Locality
           +
    Core Locality
           ↓
    Lower Memory / Communication Cost
           ↓
    Lower Makespan

---

# 33. 最终分析原则

Q3 消融实验最终不应该只得出：

    “L2 可以提高性能。”

真正需要证明的是三个层次：

第一层：

    L2 本身是否减少 DDR 访问？

第二层：

    调度算法是否能够主动提高 L2 的有效复用？

第三层：

    Cache Locality 是否能够与
    Critical Path、Core Locality 和 Parallelism
    协同优化？

因此最终论文中的 Q3 消融分析建议围绕：

    Algorithm
        ↓
    Reuse Distance
        ↓
    L2 Byte Hit Rate
        ↓
    DDR Traffic
        ↓
    Makespan

这条完整链路展开。

不要将：

    Higher Cache Hit Rate

直接等价为：

    Better Scheduling

最终评价目标仍然是 Makespan。