按照我们现在这条路线，语义分块最好不要设计成“先按 op 类型硬编码一堆规则，然后直接融合”。更稳妥的是把它做成一个三阶段过程：

**节点语义标注 → 局部模式识别 → 候选块合并/切分。**

这样既能利用高层算子知识，又不会因为误识别一个模式把整张图切坏。

先说最核心的定义。对原始计算图

$$
G=(V,E)
$$

不要直接输出 partition，而是先给每个节点构造一个语义特征向量：

$$
\phi(v)=
[
op,\ pipe,\ cycles,\ in/out\ degree,\ buffer\ usage,\ tensor\ size,\ producer/consumer\ pattern
]
$$

然后定义节点角色：

$$
role(v)\in
\{
Compute,\ Elementwise,\ Reduction,\ Communication,\ Memory,\ Control
\}.
$$

这个分类要比“高维/非高维”更实用。比如我们前面讨论的：

* MATMUL、CONV → Compute；
* ADD、MUL、RELU、SIGMOID → Elementwise；
* REDUCE、MAX、SUM → Reduction；
* COPY_IN、COPY_OUT、MOVE → Communication；
* ALLOC、FREE → Memory。

这一步的作用不是决定最终分块，而是给后续 pattern matcher 一个统一语义层。

然后第二步做“局部语义模式识别”。这里我建议不要一开始追求识别整个 Matmul 或整个 Attention，而是先识别稳定的小 motif。

例如最基础的几类：

$$
Compute \rightarrow Elementwise^*
$$

比如：

$$
MATMUL\rightarrow ADD\rightarrow RELU
$$

可以识别为：

$$
ComputeFusionBlock.
$$

再比如：

$$
Reduction
\rightarrow Elementwise
\rightarrow Reduction
$$

可能是某种归约链。

Softmax 类可以定义成更具体的模板：

$$
REDUCE\_MAX
\rightarrow SUB
\rightarrow EXP
\rightarrow REDUCE\_SUM
\rightarrow DIV
$$

识别为：

$$
SoftmaxBlock.
$$

Matmul tile 则不一定靠单个 MMAD 节点识别，而可以利用 pipe 和数据搬运模式：

$$
COPY/MOVE
\rightarrow CubeCompute
\rightarrow COPY/MOVE.
$$

25 年优秀论文里其实已经把 Matmul 的典型执行拆成 COPY_IN → MOVE → MMAD → COPY_OUT，并讨论不同 block 顺序对复用与流水的影响，这说明“局部阶段模式”是一个很自然的抽象层。

第三步才是真正做 partition。

这里最重要的一点是：

> “识别到一个语义模式” ≠ “一定全部融合成一个 partition”。

要给每个候选块一个 merge score。

我建议用：

$$
Score(A,B)
=
\lambda_1 CommSaved
+
\lambda_2 LocalityGain
+
\lambda_3 LifetimeReduction
+
\lambda_4 SemanticAffinity
-
\lambda_5 ParallelismLoss
-
\lambda_6 ResourceConflict.
$$

其中几项很好理解。

`CommSaved`：A、B 合并后能省多少跨块数据传输。

$$
CommSaved(A,B)
=
\sum_{e\in E(A,B)} size(e)
$$

`LifetimeReduction`：合并以后，某些中间 Tensor 能否更早释放。

`SemanticAffinity`：例如 Matmul→Bias→Activation 很高；两个不相关分支很低。

`ParallelismLoss`：合并以后是否丢掉原本可并行的两个块。

`ResourceConflict`：如果一个块长期占 Cube，另一个长期占 Vector，合起来未必坏；但如果两个阶段都争同一资源，块太大会恶化流水。

所以最终不是“op 连着就融合”，而是：

$$
merge(A,B)
\iff
Score(A,B)>\theta.
$$

这个设计和 25 年题解的共同经验很一致：它们都说明单纯追求缓存/搬运最小，会损害并行性；问题三必须在执行时间与数据搬运之间权衡。第一篇甚至出现了降低搬运但执行时间变差的案例。

语义分块本身我建议分四级，不要一步到位。

第一级是 `AtomicRole`，单节点语义：

```python
MATMUL -> COMPUTE
ADD -> ELEMENTWISE
REDUCE -> REDUCTION
COPY_IN -> COMMUNICATION
```

第二级是 `MicroPattern`，短链 motif：

```text
MATMUL -> ADD -> RELU
REDUCE -> EXP -> REDUCE
MOVE -> MMAD -> MOVE
```

第三级是 `StageBlock`：

```text
MatmulTile
SoftmaxBlock
ConvTile
ElementwiseChain
ReductionStage
```

第四级才是 `Partition`，用于多 Core 调度。

关系可以写成：

$$
Node
\rightarrow MicroPattern
\rightarrow StageBlock
\rightarrow Partition.
$$

这样非常重要，因为“语义块”和“调度块”不必一一对应。

例如：

```text
MatmulTile_1
MatmulTile_2
MatmulTile_3
```

语义上是三个 MatmulTile，但如果它们独立可并行，就不应该融合成一个 partition。

反过来：

```text
MatmulTile
   ↓
Bias
   ↓
Relu
```

虽然是多个语义单元，但为了减少中间 Tensor 写回，可能应该形成一个 partition。

还有一个很关键的边界规则：**fan-out 和 fan-in 点天然是候选切分点。**

如果：

$$
outdegree(v)>1
$$

说明这个结果被多个分支消费，贸然往下融合会复制依赖或压缩并行空间。

类似地：

$$
indegree(v)>1
$$

说明这是汇合点，也应该谨慎融合。

所以可以定义 boundary score：

$$
Boundary(v)
=
\alpha\,fanout(v)
+\beta\,fanin(v)
+\gamma\,tensor\_size(v)
+\delta\,resource\_switch(v).
$$

其中 `resource_switch` 指 pipe 从 Cube 切到 Vector、MTE 等。

高 BoundaryScore 的位置优先切。

这样就可以把“高维知识”和图结构结合起来，而不是只靠 op 名字。

实现上，我建议数据结构直接这样设计：

```python
@dataclass
class NodeFeature:
    id: int
    op: str
    pipe: str
    cycles: int

    role: str

    indegree: int
    outdegree: int

    input_bufs: list[int]
    output_bufs: list[int]

    compute_weight: float
    memory_weight: float
    criticality: float


@dataclass
class SemanticBlock:
    id: int
    nodes: list[int]

    semantic_type: str

    dominant_pipe: str

    cycles: int

    input_tensors: set[int]
    output_tensors: set[int]

    internal_comm: float
    external_comm: float

    buffer_pressure: float
    pipelineability: float

    predecessors: set[int]
    successors: set[int]
```

然后算法框架可以非常明确：

```python
def semantic_partition(graph):

    # 1. 节点特征
    features = extract_features(graph)

    # 2. 节点语义
    roles = classify_roles(features)

    # 3. motif识别
    motifs = detect_patterns(graph, roles)

    # 4. 初始语义块
    blocks = build_stage_blocks(motifs)

    # 5. 计算块间代价
    scores = evaluate_merge_scores(blocks)

    # 6. hardware-aware merge
    blocks = greedy_merge(blocks, scores)

    # 7. split oversized / bad blocks
    blocks = refine_split(blocks)

    # 8. 构建Partition DAG
    return build_partition_dag(blocks)
```

我甚至建议初版先不要上复杂机器学习，直接做“规则 + 图结构 + 打分”。

比如 `detect_patterns()` 第一版只识别 4～5 类：

```text
1. Compute + Elementwise chain
2. Reduction chain
3. Matmul/Cube stage
4. Communication + Compute stage
5. Repeated isomorphic motif
```

然后逐步加。

重复结构识别也很重要。比如两个子图拥有相同的：

$$
op\ sequence
$$

和近似的：

$$
degree\ pattern,\ pipe\ pattern,\ buffer\ pattern
$$

就可以构造 hash：

$$
H(P)
=
hash(
op\ sequence,
pipe\ sequence,
degree\ signature
).
$$

相同 hash 的子图可以判为同类模板。

这样以后会非常好用，因为你可以给同一模板复用相同 partition 策略。

例如：

```text
MatmulTile#1
MatmulTile#2
MatmulTile#3
...
```

不用每个重新探索。

这里我建议再加一个非常实用的概念：

$$
pipelineability(P)
$$

用于判断一个块到底适不适合做流水。

可以定义成：

$$
Pipelineability(P)
=
\frac{
ResourceDiversity(P)\cdot ParallelSlack(P)
}{
CriticalSerialLength(P)
}.
$$

不用一开始追求特别严格，先做归一化 heuristic 即可。

直觉上：

* MTE → Cube → MTE 类型块，pipelineability 高；
* 全部都是 Cube 串行依赖，低；
* 大 fan-out 分支，高并行潜力；
* 一条深长 reduction chain，低。

这个值会直接影响是否合并。

最后我特别建议：语义分块不要直接输出“唯一答案”，而是输出**候选 partition 集**。

比如：

$$
\mathcal P^{(1)},\mathcal P^{(2)},\mathcal P^{(3)}
$$

分别偏：

* 通信最小；
* 并行度最高；
* buffer 压力最小。

然后后面的 Multi-Core Scheduler 用 simulator 评估：

$$
T(\mathcal P)
$$

最后选择真正表现好的方案。

这个和 25 年多初始解思想是一脉相承的：第一篇就用了贪心、DFS、关键路径三种不同偏好的初始序列再继续优化。

所以我觉得最稳的一版语义分块设计，可以浓缩成：

$$
\boxed{
\text{Role labeling}
\rightarrow
\text{Motif detection}
\rightarrow
\text{Semantic block}
\rightarrow
\text{Hardware-aware merge/split}
\rightarrow
\text{Candidate partitions}
}
$$

其中真正的创新点不在于“识别 MATMUL”，而是：

$$
\boxed{
SemanticAffinity
+
Communication
+
Lifetime
+
Parallelism
+
Resource
}
$$

共同决定 partition。

如果现在要开始落代码，我会先实现最小可用版本：`NodeFeatureExtractor → RoleClassifier → PatternMatcher → BlockBuilder → MergeScorer → PartitionDAG`。这六个模块已经足够支撑后面的多 Core scheduler，而且每个模块都可以独立测试。
