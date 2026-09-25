> 文档定位：本文是 Python 模块、数据结构和代码骨架草案。统一算法总纲见 [`demo_algorithm.md`](demo_algorithm.md)；题目固定参数和评估器行为以 `challenge_*` 文档为准。
>
> 工程边界：可以保留快速代理 simulator 用于筛选候选，但官方 evaluator 才是最终性能真值；算法输出仍只包含切图、分核和同核子图顺序。

可以把目前的思路收敛成一条比较清晰的主线：

> 原始计算图 → 图语义/结构特征提取 → 面向流水与并行的图划分 → 显式构造 partition 间通信与同步 → 多 Core 调度 → 仿真评估 → 局部搜索/迭代优化。

三个问题的区别，不建议设计成三套完全独立算法，而是做成一个统一框架，随着题目递进逐步打开约束和优化变量。这样代码实现、论文建模和后续调参都会顺很多。

---

## 1. 统一数学抽象

首先把原始计算图记成 DAG：

$$
G=(V,E)
$$

其中每个算子节点

$$
v_i\in V
$$

至少具有：

$$
v_i=
(op_i,c_i,p_i,r_i,\mathcal I_i,\mathcal O_i)
$$

其中：

* \(op_i\)：算子类型，如 MATMUL、ADD、REDUCE、COPY_IN；
* \(c_i\)：执行 cycle；
* \(p_i\)：pipe / 执行单元类别；
* \(r_i\)：资源需求；
* \(\mathcal I_i,\mathcal O_i\)：输入输出 Tensor。

Tensor 边：

$$
e_{ij}=(v_i,v_j,t_{ij})
$$

其主要属性可以定义为：

$$
t_{ij}=(s_{ij},pos_{ij},type_{ij})
$$

其中 \(s_{ij}\) 是 Tensor size。

后续所有算法实际上是在求三个东西：

$$
\boxed{
\text{Partition}
+
\text{Placement}
+
\text{Schedule}
}
$$

即：

1. 哪些算子归为一个 partition；
2. partition 放在哪个 Core；
3. partition 什么时候开始执行。

---

# 2. 第一层：计算图高层语义提取

这一步不要把“高维知识”理解成一定要推断真实 tensor rank。

题目数据里没有完整 shape，所以实际可以定义成：

> 根据算子类型、数据流模式以及结构上下文推断出的计算语义标签。

推荐形成这样的节点标签。

```text
MATMUL / CONV
    -> DENSE_COMPUTE

REDUCE
    -> REDUCTION

ADD / MUL / SUB / DIV
    -> ELEMENTWISE

RELU / SIGMOID / EXP / SQRT
    -> ACTIVATION_OR_ELEMENTWISE

COPY_IN / COPY_OUT
    -> COMMUNICATION
```

此外，再为每个节点提取结构特征：

$$
f(v)=
[
cycle,
tensor\_size,
indegree,
outdegree,
depth,
criticality,
op\_class,
pipeline,
reuse
]
$$

比较重要的几个特征：

### 2.1 DAG 层级

```python
depth[v]
```

表示从源点到当前节点的拓扑深度。

可以识别：

```text
Layer 0   COPY_IN
             |
Layer 1   MATMUL
          /    \
Layer 2 ADD    ADD
        |       |
Layer 3 RELU   RELU
```

后面天然可以用来识别并行支路。

---

### 2.2 关键路径

对每个节点定义 bottom level：

$$
b(v_i)=c_i+
\max_{v_j\in succ(i)}
b(v_j)
$$

没有后继时：

$$
b(v_i)=c_i
$$

它表示：

> 如果从当前节点开始一路走到 DAG 终点，剩余的最长执行时间。

调度阶段优先处理 \(b(v)\) 大的节点。

这基本是 HEFT / critical-path-aware scheduler 的核心思想。

---

### 2.3 重复路径识别

这是目前比较值得做的一块。

例如：

```text
MATMUL -> ADD -> RELU
MATMUL -> ADD -> RELU
MATMUL -> ADD -> RELU
MATMUL -> ADD -> RELU
```

显然不是四条完全独立的随机路径，而可能代表：

```text
多头结构
多通道结构
block replication
```

可以给节点建立局部结构签名：

```python
signature(v) = (
    op_class,
    child_op_classes,
    parent_op_classes,
    depth,
)
```

更进一步，对长度 \(k\) 的子路径：

```python
MATMUL -> ADD -> RELU
```

编码为：

```python
("DENSE", "ELEMENTWISE", "ACTIVATION")
```

再做频繁模式匹配。

这个信息后面可以用于：

* 对称 partition；
* 对称 Core 分配；
* 负载均衡；
* 同构流水线。

---

# 3. 第二层：候选融合 / region 构造

这一阶段的思想不是马上决定 partition，而是先识别：

> 哪些节点比较适合待在一起。

例如：

```text
MATMUL
   |
 ADD
   |
 RELU
```

可以成为：

```text
FusionRegion #3
```

推荐采用“anchor + attachable op”的思路。

Anchor：

```text
MATMUL
CONV
REDUCE
```

Attachable：

```text
ADD
MUL
RELU
SIGMOID
EXP
DIV
SQRT
```

但并不是所有 attachable 都应该融合。

建议定义融合收益：

$$
Gain(i,j)
=
\alpha C_{comm}(i,j)
+
\beta C_{sync}(i,j)
-
\gamma C_{resource}(i,j)
-
\delta C_{critical}(i,j)
$$

直观来说：

把两个节点放一起的收益包括：

* Tensor 不需要跨 partition；
* 少一次同步；
* 减少 buffer；
* 增加局部性。

代价包括：

* partition 太大；
* pipeline 太单一；
* 破坏并行度；
* 导致 Core 负载不均。

因此融合条件可以写成：

$$
Gain(i,j)>0
$$

而不是简单：

```python
if predecessor is MATMUL:
    merge()
```

---

# 4. 第三层：Partition 模型

这是整个模型最核心的一层。

定义：

$$
P=\{P_1,P_2,\cdots,P_K\}
$$

满足：

$$
\bigcup_k P_k=V
$$

并且：

$$
P_i\cap P_j=\emptyset
$$

每个 partition 应该尽量具有：

1. 强内部依赖；
2. 较低跨 partition 通信；
3. 可流水执行；
4. 不破坏明显的并行结构；
5. 单个 partition 资源不过大。

---

## 4.1 Partition 的目标函数

可以统一写成：

$$
\min
F(P)
=
\alpha C_{\text{cut}}
+
\beta C_{\text{imbalance}}
+
\gamma C_{\text{critical}}
+
\delta C_{\text{resource}}
-
\eta B_{\text{pipeline}}
$$

其中：

### Cut communication

$$
C_{\text{cut}}
=
\sum_{(u,v)\in E}
size_{uv}
\cdot
\mathbf 1[P(u)\neq P(v)]
$$

这个应该是划分阶段最基础的项。

---

### Partition 负载均衡

设：

$$
W(P_k)=\sum_{v\in P_k}c_v
$$

则：

$$
C_{\text{imbalance}}
=
\max_kW(P_k)
-
\frac1K\sum_kW(P_k)
$$

或者：

$$
Var(W(P_k))
$$

---

### 关键路径破坏

如果关键路径被切得太碎：

```text
MATMUL
 |
ADD
 |
RELU
 |
MUL
```

每一步之间都需要同步，会非常亏。

因此：

$$
C_{\text{critical}}
=
\sum_{(u,v)\in E_{critical}}
w_{uv}
\mathbf 1[P(u)\neq P(v)]
$$

对关键路径上的边提高 cut penalty。

---

### Pipeline 奖励

如果 partition 内部出现：

```text
PIPE_A -> PIPE_B -> PIPE_A -> PIPE_B
```

通常比：

```text
PIPE_A -> PIPE_A -> PIPE_A
```

更有机会实现流水重叠。

可以定义：

$$
B_{\text{pipeline}}
=
\sum_{(u,v)\in E,\ P(u)=P(v)}
\mathbf 1[pipe(u)\neq pipe(v)]
$$

这是一个粗模型，后期可以被真实性能模拟器替代。

---

# 5. 不建议直接使用通用图划分器作为主算法

Metis 这种图划分算法可以当 baseline，但这里问题不是简单 min-cut。

因为这个问题本质是：

> DAG + heterogeneous operator + pipeline + buffer + synchronization + multi-core scheduling

所以最合适的是：

```text
结构规则初始化
        ↓
cost-based merge/split
        ↓
scheduler-in-the-loop refinement
```

也就是说：

> partition 好不好，最终要看调度完成后的 makespan，而不只是 cut cost。

---

# 6. 第四层：Partition Graph

节点融合完成后，把原 DAG 压缩：

$$
G_P=(P,E_P)
$$

例如：

```text
v0 -> v1 -> v2 -> v3
       \
        -> v4
```

划分：

```text
P0 = {v0}
P1 = {v1,v2}
P2 = {v3}
P3 = {v4}
```

得到：

```text
       P2
      /
P0 -> P1
      \
       P3
```

这一步非常重要。

后续 scheduler **不应该继续直接调原始 op**。

而是：

```text
graph scheduler
        ↓
partition scheduler
        ↓
partition 内局部 scheduler
```

形成两级调度。

---

# 7. 第五层：显式 Buffer / 同步模型

一旦产生 partition：

```text
P_i -> P_j
```

并且：

$$
core(P_i)\neq core(P_j)
$$

则建立一个：

```python
Buffer
```

例如：

```python
Buffer(
    src_partition=3,
    dst_partition=7,
    tensor="tensor_123",
    size=8192
)
```

同步关系：

```text
P3 finish compute
       ↓
write buffer
       ↓
buffer ready
       ↓
P7 read buffer
       ↓
P7 compute
```

所以调度约束不是简单：

$$
S_j\ge F_i
$$

而是：

$$
S_j
\ge
F_i+C_{comm}(i,j)
$$

如果有 buffer 冲突，还要：

$$
S_j
\ge
T_{\text{buffer available}}
$$

---

# 8. 第六层：多 Core 调度

对于每个 partition \(P_i\) 和 Core \(c\)，定义：

$$
S_{ic}
$$

开始时间。

实际代码里没必要直接解 MILP，先用启发式。

最推荐的第一版：

> Critical-Path-Aware List Scheduling

优先级：

$$
Priority(P_i)
=
\alpha Rank_u(P_i)
+
\beta CP(P_i)
+
\gamma OutDegree(P_i)
+
\delta DataReuse(P_i)
$$

其中最重要的是：

$$
Rank_u
$$

即从该 partition 到终点的最长预计时间。

调度过程：

```python
ready_queue = 所有依赖已完成的 partition

while ready_queue:
    p = highest_priority()

    for core in cores:
        EST = earliest_start_time(p, core)
        EFT = EST + exec_time(p, core)

    core = argmin(EFT)

    assign(p, core)
```

因为题目里 Core 同构，所以实际上：

$$
exec(P,c_1)=exec(P,c_2)
$$

这会比标准 HEFT 简化很多。

主要差异来自：

```text
Core availability
communication
buffer
pipeline contention
```

而不是 Core 算力差异。

---

# 9. 调度时必须进一步考虑 Core 内 pipeline

这是这个题和普通 DAG scheduling 区别比较大的地方。

一个 Core 内可能有：

```text
PIPE0
PIPE1
PIPE2
```

如果：

```text
P1 使用 PIPE0
P2 使用 PIPE1
```

它们理论上可能产生重叠。

因此 Core 不能只维护：

```python
core.available_time
```

最好维护：

```python
core.pipeline_available = {
    PIPE0: t0,
    PIPE1: t1,
    PIPE2: t2,
}
```

节点 \(v\) 的最早启动：

$$
EST(v)=
\max
\begin{cases}
dependency\_ready(v)\\
pipe\_available[pipe(v)]\\
buffer\_ready(v)
\end{cases}
$$

这会直接形成一个事件驱动模拟器。

---

# 10. 整体 Python 工程结构

推荐不要一开始写成一个 `main.py` 3000 行。

可以直接按下面拆。

```text
src/
│
├── model/
│   ├── graph.py
│   ├── operator.py
│   ├── tensor.py
│   ├── partition.py
│   ├── hardware.py
│   └── schedule.py
│
├── parser/
│   └── case_parser.py
│
├── analysis/
│   ├── graph_feature.py
│   ├── semantic.py
│   ├── critical_path.py
│   ├── motif.py
│   └── visualization.py
│
├── partition/
│   ├── initial_partition.py
│   ├── fusion.py
│   ├── split.py
│   ├── cost.py
│   └── optimizer.py
│
├── scheduler/
│   ├── priority.py
│   ├── core_scheduler.py
│   ├── pipeline_scheduler.py
│   └── scheduler.py
│
├── simulator/
│   ├── event.py
│   ├── simulator.py
│   └── metrics.py
│
├── optimize/
│   ├── local_search.py
│   ├── annealing.py
│   └── search.py
│
├── experiment/
│   ├── baseline.py
│   └── benchmark.py
│
└── main.py
```

核心原则是：

```text
算法
和
性能仿真器
彻底分离
```

后面会非常省事。

---

# 11. 最底层的数据结构

例如：

```python
from dataclasses import dataclass, field
from typing import List, Dict, Set


@dataclass
class Tensor:
    id: str
    size: int
    pos: str | None = None


@dataclass
class Operator:
    id: int
    op_type: str
    cycles: int
    pipe: str

    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

    predecessors: Set[int] = field(default_factory=set)
    successors: Set[int] = field(default_factory=set)

    # semantic features
    op_class: str | None = None
    depth: int = 0
    rank_u: float = 0.0
    critical: bool = False
```

图：

```python
@dataclass
class ComputeGraph:
    operators: Dict[int, Operator]
    tensors: Dict[str, Tensor]

    topo_order: List[int] = field(default_factory=list)
```

---

# 12. Partition 数据结构

```python
@dataclass
class Partition:
    id: int

    operators: List[int]

    predecessors: Set[int] = field(default_factory=set)
    successors: Set[int] = field(default_factory=set)

    total_cycles: int = 0

    input_tensors: Set[str] = field(default_factory=set)
    output_tensors: Set[str] = field(default_factory=set)

    critical_rank: float = 0.0

    pipeline_cycles: Dict[str, int] = field(default_factory=dict)
```

后面甚至可以定义：

```python
@property
def intensity(self):
    return self.total_cycles / self.io_bytes
```

---

# 13. Hardware Model

题目硬件最好单独抽象。

```python
@dataclass
class Core:
    id: int
    pipelines: List[str]


@dataclass
class Hardware:
    num_cores: int
    cores: List[Core]

    bandwidth: float
    sync_latency: int
```

后面不同问题改硬件约束，只改这里。

---

# 14. FeatureExtractor

建议所有“高维知识”集中在这一层。

```python
class GraphFeatureExtractor:

    def run(self, graph):
        self.infer_semantics(graph)
        self.compute_depth(graph)
        self.compute_rank_u(graph)
        self.find_critical_path(graph)
        self.detect_parallel_branches(graph)
        self.detect_repeated_patterns(graph)

        return graph
```

语义：

```python
def classify_op(op_type):
    if op_type in {"MATMUL", "CONV"}:
        return "DENSE_COMPUTE"

    if op_type == "REDUCE":
        return "REDUCTION"

    if op_type in {
        "ADD", "MUL", "DIV", "SUB",
        "RELU", "SIGMOID", "EXP", "SQRT"
    }:
        return "ELEMENTWISE"

    if op_type.startswith("COPY"):
        return "COMMUNICATION"

    return "OTHER"
```

---

# 15. Critical Path 实现

```python
def compute_rank_u(graph):
    rank = {}

    for v in reversed(graph.topo_order):

        op = graph.operators[v]

        if not op.successors:
            rank[v] = op.cycles
        else:
            rank[v] = (
                op.cycles
                + max(rank[u] for u in op.successors)
            )

        op.rank_u = rank[v]

    return rank
```

如果加入通信：

```python
rank[v] = cycles[v] + max(
    comm_cost(v, u) + rank[u]
    for u in succ[v]
)
```

这已经基本够第一版 scheduler 用。

---

# 16. 初始 Partition 算法

建议第一版不要贪复杂。

可以：

```text
Step 1
每个 anchor 算子独立创建 partition

Step 2
沿 DAG 向后吸收 elementwise

Step 3
遇到以下情况停止：

branch
merge
REDUCE
COPY
resource threshold
critical parallel structure
```

伪代码：

```python
for node in topo_order:

    if node already assigned:
        continue

    if is_anchor(node):

        p = new_partition(node)

        cur = node

        while True:

            successors = cur.successors

            if len(successors) != 1:
                break

            nxt = only_successor(cur)

            if indegree(nxt) != 1:
                break

            if not is_fusable(nxt):
                break

            if not resource_ok(p, nxt):
                break

            p.add(nxt)
            cur = nxt
```

这个作为 baseline 会非常稳定。

---

# 17. 再做 Cost-based Merge

初始化以后，对两个相邻 partition：

$$
P_i,P_j
$$

计算：

```python
delta = cost(merge(Pi, Pj)) - cost(Pi, Pj)
```

如果：

```python
delta < 0
```

则 merge。

但这里最好不要只算静态 cost。

后期升级成：

```python
old_score = simulator(partitions)
new_score = simulator(merged_partitions)

if new_score < old_score:
    accept
```

即：

> scheduler-in-the-loop。

这是后期真正容易出效果的地方。

---

# 18. Scheduler 框架

建议接口做成：

```python
class Scheduler:

    def schedule(
        self,
        graph,
        partitions,
        hardware
    ):
        ...
```

返回：

```python
ScheduleResult(
    makespan=...,
    placements=...,
    events=...
)
```

调度主体：

```python
while unscheduled:

    ready = [
        p for p in unscheduled
        if all(dep in scheduled for dep in p.predecessors)
    ]

    ready.sort(
        key=lambda p: p.critical_rank,
        reverse=True
    )

    p = ready[0]

    best = None

    for core in hardware.cores:

        start = earliest_start(
            p,
            core,
            state
        )

        finish = estimate_finish(
            p,
            core,
            start
        )

        if best is None or finish < best.finish:
            best = Candidate(core, start, finish)

    commit(p, best)
```

---

# 19. 事件驱动 Simulator

后面真正重要的是 simulator。

推荐 Schedule 输出不要只是：

```python
partition -> core
```

而是事件。

例如：

```python
Event(
    type="COMPUTE",
    core=0,
    pipe="MATRIX",
    start=100,
    end=180,
    partition=3
)
```

以及：

```python
Event(
    type="TRANSFER",
    src_core=0,
    dst_core=2,
    start=180,
    end=210,
    tensor="t32"
)
```

这样可以直接画 Gantt Chart。

最终优化目标：

$$
Makespan
=
\max_e end(e)
$$

---

# 20. Metrics 统一收集

至少输出：

```python
@dataclass
class Metrics:

    makespan: int

    communication_cycles: int
    synchronization_cycles: int

    core_utilization: dict
    pipeline_utilization: dict

    load_balance: float

    num_partitions: int
    num_cross_partition_edges: int

    total_buffer_bytes: int
```

后面实验特别有用。

不然你只看到：

```text
score = 18923
```

完全不知道算法为什么变好或变差。

---

# 21. 三个题目在同一框架里的演化

我更推荐这样理解题目的递进。

### 问题 1：建立可行 baseline

重点：

```text
DAG
↓
partition
↓
Core placement
↓
合法 schedule
```

算法可以相对简单：

```text
语义规则融合
+
greedy partition
+
critical-path list scheduling
```

模型目标：

$$
\min Makespan
$$

先不追求非常复杂的 buffer / search。

---

### 问题 2：把通信和流水显式纳入

扩展：

```text
partition edge
↓
buffer
↓
communication
↓
synchronization
```

调度模型从：

$$
S_j\ge F_i
$$

变成：

$$
S_j
\ge
F_i+Comm_{ij}+Sync_{ij}
$$

此时图划分目标也变成：

$$
partition
\leftrightarrow
schedule
$$

联合优化。

这一问开始，“切得均匀”未必最好。

---

### 问题 3：联合优化

最终版应该是：

$$
\min_{P,M,S}
Makespan(P,M,S)
$$

其中：

* \(P\)：partition；
* \(M\)：core mapping；
* \(S\)：schedule。

这是组合优化问题。

不建议直接精确求。

使用：

```text
规则/贪心给初值
        ↓
critical-path scheduling
        ↓
performance simulation
        ↓
local search
        ↓
repartition / remap / reschedule
```

---

# 22. Local Search 应该搜索什么

真正实现时，搜索空间不应该是“随便改变整个 DAG”。

定义几个 move 就够了。

例如：

```python
MoveOperatorToNeighborPartition
MergeTwoPartitions
SplitPartition
SwapCore
MovePartitionToCore
```

每次产生 candidate：

```python
candidate = apply_move(solution)

score = simulator(candidate)
```

接受：

```python
if score < best_score:
    accept
```

后面可以换成 simulated annealing：

$$
P(\text{accept})
=
e^{-\Delta/T}
$$

避免陷入局部最优。

---

# 23. 一个比较完整的 Python 主流程

最后整个程序可以长成这样：

```python
def solve(case_file):

    # ---------------------------------
    # 1. parse
    # ---------------------------------

    graph = CaseParser().load(case_file)

    # ---------------------------------
    # 2. graph analysis
    # ---------------------------------

    extractor = GraphFeatureExtractor()

    extractor.run(graph)

    # ---------------------------------
    # 3. initial partition
    # ---------------------------------

    partitioner = InitialPartitioner()

    partitions = partitioner.partition(graph)

    # ---------------------------------
    # 4. partition refinement
    # ---------------------------------

    partition_optimizer = PartitionOptimizer()

    partitions = partition_optimizer.optimize(
        graph,
        partitions
    )

    # ---------------------------------
    # 5. build partition DAG
    # ---------------------------------

    pgraph = build_partition_graph(
        graph,
        partitions
    )

    # ---------------------------------
    # 6. hardware
    # ---------------------------------

    hardware = load_hardware_config()

    # ---------------------------------
    # 7. scheduling
    # ---------------------------------

    scheduler = CriticalPathScheduler()

    schedule = scheduler.schedule(
        pgraph,
        hardware
    )

    # ---------------------------------
    # 8. simulation
    # ---------------------------------

    simulator = Simulator(hardware)

    metrics = simulator.run(
        graph,
        pgraph,
        schedule
    )

    # ---------------------------------
    # 9. joint optimization
    # ---------------------------------

    optimizer = LocalSearchOptimizer(
        simulator=simulator,
        scheduler=scheduler
    )

    solution = optimizer.optimize(
        graph,
        partitions,
        schedule
    )

    return solution
```

这基本就是最后比赛代码的骨架。

---

# 24. 我建议当前先实现的版本

不要一次把所有东西铺开。

现在比较合理的 V1 是：

```text
JSON parser
    ↓
NetworkX DAG
    ↓
算子语义分类
    ↓
depth / topo / critical path
    ↓
fusion-region detection
    ↓
greedy partition
    ↓
partition DAG
    ↓
critical-path-aware list scheduling
    ↓
Gantt 可视化
```

先做到：

> 给任何一个 case，都能够把“原始图 → partition → Core → 时间轴”完整跑通。

然后 V2：

```text
加入通信代价
加入 buffer
加入 pipeline overlap
```

最后 V3：

```text
local search
repartition
core remapping
scheduler-in-the-loop
```

---

整体上，现在这套模型可以浓缩成一句比较适合写进论文的方法论：

$$
\boxed{
\text{Semantic-aware Graph Partitioning}
+
\text{Critical-path-aware Multi-core Scheduling}
+
\text{Simulation-guided Joint Optimization}
}
$$

也就是：

> 利用算子语义、重复子图、关键路径等高层结构知识生成适于流水和并行的计算子图；将跨子图 Tensor 显式建模为通信与同步依赖；随后采用关键路径驱动的多 Core 列表调度，并通过事件级性能模拟反馈调整图划分与 Core 映射，最终实现划分—映射—调度的联合优化。

这个表述已经基本能作为你们后面“模型设计 + 算法设计 + Python 工程实现”的统一主干了。
