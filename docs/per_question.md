对。现在这版算法更像是“一个统一算法 + `scenario` 参数”，但题目的三问其实会改变**分块收益函数、任务粒度、Core 亲和性以及调度评价函数**，不应该只在调度时改一个通信常数。

从当前代码能明确确认的一点是：程序实际上只区分了

$$
Q1 \quad\text{和}\quad \{Q2,Q3\}
$$

两种代价模型。`_schedule_partitions()` 中，Q1 使用“同 Core 等待 100、跨 Core 等待 1000，再加传输时间”，而 Q2/Q3 都进入同一个分支：只有跨 Core 才增加 \(500+\lceil bytes/60\rceil\)。

所以现在真正需要做的是：让“三问条件 → 算法行为”发生系统性变化。

---

## 一、先看最核心的影响：三问会改变“什么才是好 partition”

现在 `semantic_partition()` 基本与 `scenario` 无关：

$$
\text{Graph}
\rightarrow
\text{semantic partition}
\rightarrow
\text{scenario-aware scheduling}
$$

也就是说，Q1/Q2/Q3 得到的 partition 基本一样，只是后面调度不同。

但从优化角度，更合理的是：

$$
\boxed{
\text{Graph}
+
\text{Question Conditions}
\rightarrow
\text{question-aware partition}
\rightarrow
\text{question-aware scheduling}
}
$$

原因很直接：

如果一个 partition 边在 Q1 中代价很大，就应该尽量通过融合消掉；

而如果在 Q2 中，同 Core 的两个 partition 之间几乎没有额外代价，那么完全没必要为了消掉这条边而提前融合——可以保留两个 partition，让 scheduler 后面决定它们是否放在同 Core。

这会直接改变最优粒度。

---

# 二、Q1 对算法的影响：更强调“减少 partition 边界”

按照当前代码采用的 Q1 模型，对于 partition 边

$$
P_i\rightarrow P_j
$$

即使两个 partition 被安排在同一个 Core，也有：

$$
d_{ij}
=
100+
\left\lceil\frac{s_{ij}}{60}\right\rceil.
$$

如果跨 Core：

$$
d_{ij}
=
1000+
\left\lceil\frac{s_{ij}}{60}\right\rceil.
$$



因此 Q1 有一个很重要的性质：

> **只要切开 partition，就可能付出代价；放在同 Core 只能降低代价，不能完全消除代价。**

于是 Q1 的分块策略应该明显偏向“融合”。

### 对分块算法的影响

现在的 merge score 是：

$$
M(A,B)
=
C_{\rm saved}
+
L
+
S
+
R
-
P_{\rm loss}
-
B
-
S_{\rm serial}.
$$

其中通信收益只考虑 payload transmission time，而且代码明确说明没有加入 fixed wait，因为当前 partitioner 是 scenario-independent。

但是 Q1 下，这个固定等待恰恰非常重要。

所以 Q1 应该把 merge communication gain 改成类似：

$$
G^{Q1}_{comm}(A,B)
=
\tau_{\rm same}
+
\left\lceil
\frac{s_{AB}}{BW}
\right\rceil ,
$$

其中

$$
\tau_{\rm same}=100.
$$

因为一旦两个 block 真正融合为一个 partition，这个 partition 边直接消失。

也就是说：

$$
M_{Q1}(A,B)
=
M_{\rm semantic}(A,B)
+
\lambda G^{Q1}_{comm}(A,B)
-
\mu P_{\rm loss}.
$$

### 结果是什么？

Q1 应该倾向于：

* Compute → Elementwise 更积极融合；
* Reduction → Elementwise 更积极融合；
* 大 Tensor 边更积极融合；
* 长链上的小 partition 尽量合并；
* 但 fan-out / fan-in 处仍谨慎，因为这里融合会损失 Core 并行性。

所以 Q1 的典型 partition 应该是：

$$
\boxed{\text{较少、较大、通信边界少}}
$$

而不是大量小任务。

---

# 三、Q1 的调度也应该更强烈地考虑“前驱 Core”

现在 Q1 scheduler 已经会算：

$$
EST(i,k)
=
\max\left\{
T_k,
\max_{j\in pred(i)}
(F_j+d_{ji}(k))
\right\}.
$$

因此跨 Core 的 1000-cycle penalty 会自然让后继跟随前驱。

但这里还可以进一步强化。

对于任务 \(i\)，可以显式定义：

$$
Affinity_{Q1}(i,k)
=
\sum_{j\in pred(i)}
\left[
900+
\left\lceil\frac{s_{ji}}{60}\right\rceil
\right]
\mathbf 1(core(j)=k).
$$

因为从别的 Core 移到前驱 Core，至少能少掉：

$$
1000-100=900
$$

的 fixed wait。

所以 Q1 的 scheduler 可以理解成：

$$
\boxed{
\text{Critical Path}
+
\text{Strong Core Affinity}
}
$$

也就是说，Q1 最怕的是：

> 为了一点负载均衡，把一条本来连续的依赖链在 Core 之间来回跳。

---

# 四、Q2 的条件会把问题性质明显改变

当前实现中，Q2 的 dependency cost 是：

同 Core：

$$
d_{ij}=0
$$

跨 Core：

$$
d_{ij}
=
500+
\left\lceil\frac{s_{ij}}{60}\right\rceil.
$$



这和 Q1 有一个本质差异：

> **Q2 中 partition 边本身不一定是坏事。**

只要两个相邻 partition 最终落在同一个 Core：

$$
P_i@C_k
\rightarrow
P_j@C_k
$$

那么它们之间的额外代价可以被隐藏/消除。

于是 Q2 下：

$$
\text{是否切 partition}
$$

和

$$
\text{partition 放哪个 Core}
$$

强耦合起来。

---

# 五、因此 Q2 不应该像 Q1 那么激进地融合

举个非常典型的情况：

$$
A\rightarrow B\rightarrow C
$$

假设：

$$
C_A=C_B=C_C=500.
$$

Q1 中如果每个 partition 边都需要至少 100 cycle，那么：

$$
A|B|C
$$

本身就产生额外开销。

所以：

$$
[A,B,C]
$$

可能比较有吸引力。

但 Q2 中，如果调度成：

$$
A@Core_0
\rightarrow B@Core_0
$$

那么：

$$
A\rightarrow B
$$

即使没有融合，也没有跨 Core 通信代价。

因此 Q2 可以保留：

$$
[A]\ [B]\ [C]
$$

这种更细粒度结构。

这样做的优势是 scheduler 仍然拥有：

$$
B@Core_0
\quad\text{或}\quad
B@Core_1
$$

的选择权。

所以：

$$
\boxed{
Q2\text{ 的 partition 应该比 Q1 更细}
}
$$

这是我认为现在代码最值得改的一点。

---

# 六、Q2 的 partition objective 应该改变

Q1 可以看作：

$$
\text{融合收益}
\approx
\text{消除 partition 边}.
$$

Q2 则应该变成：

$$
\text{融合收益}
\approx
\text{消除“未来可能跨 Core”的 partition 边}.
$$

也就是说，不能把所有 Tensor 边都认为具有同样高的融合收益。

推荐定义：

$$
M_{Q2}(A,B)
=
S(A,B)
+
\lambda C_{AB}
-
\mu P_{loss}
-
\nu F_{loss}.
$$

这里增加一个非常重要的项：

$$
F_{loss}
$$

表示：

> **合并以后丧失的调度自由度。**

例如两个大 compute block：

$$
MATMUL_1\rightarrow MATMUL_2
$$

即使中间 Tensor 很大，也不能简单融合。

因为：

$$
MATMUL_1,\ MATMUL_2
$$

保留为不同 partition 后，调度器仍然可以根据 Core 空闲情况进行放置。

而你们现在 `_merge_score()` 已经通过：

$$
parallelism\_loss
$$

和 `boundary_score` 在做这件事的初步近似。

Q2 应该进一步提高这类 parallelism penalty。

所以大致应该：

$$
\lambda_{\rm comm}^{Q1}
>
\lambda_{\rm comm}^{Q2}
$$

而：

$$
\lambda_{\rm parallel}^{Q2}
>
\lambda_{\rm parallel}^{Q1}.
$$

---

# 七、Q2 真正应该加强的是 scheduler，而不是 partitioner

Q2 里最有价值的是：

$$
\boxed{\text{Same-Core Tensor Reuse}}
$$

因此 scheduler 应显式维护每个 partition 的数据亲和关系。

定义：

$$
D(i,k)
=
\sum_{j\in pred(i),core(j)=k}s_{ji}.
$$

它表示：

> partition \(i\) 如果放到 Core \(k\)，有多少输入 Tensor 已经天然位于这个 Core。

然后：

$$
CrossBytes(i,k)
=
\sum_{j\in pred(i)}
s_{ji}
\mathbf1[core(j)\neq k].
$$

所以候选 Core 的代价可以直接写成：

$$
Score_{Q2}(i,k)
=
EFT(i,k)
+
\lambda CrossBytes(i,k).
$$

或者更物理一点：

$$
Score_{Q2}(i,k)
=
\max
\left(
EFT,
\frac{DDRBytes+CrossBytes(i,k)}{BW}
\right).
$$

其实你们现在已经有这个模型的雏形：`estimated_ddr_bytes` 与 `candidate_edge_bytes` 已经被放入 DDR lower bound。

所以 Q2 最应该加强的是：

> **Tensor-aware Core Placement**

而不是继续狂加融合规则。

---

# 八、Q2 对复杂图尤其意味着“保持主链 Core 连续性”

Attention / residual / reduction join 中，经常出现：

$$
A\rightarrow B\rightarrow C\rightarrow D
$$

以及：

$$
X\rightarrow D.
$$

如果关键路径变成：

$$
A@0\rightarrow B@1\rightarrow C@0\rightarrow D@2
$$

那就会产生大量跨 Core Tensor movement。

因此你们现在的：

$$
critical\ path\ sticky
$$

思路其实特别适合 Q2。

当前代码会找一条考虑 partition cycles 和 communication 的 heavy downstream path。

然后优先让关键前驱和当前 partition 留在同一 Core。

所以 Q2 可以明确形成：

$$
\boxed{
Critical\ Path
+
Tensor\ Affinity
+
Load\ Balance
}
$$

三个目标。

---

# 九、Q3 是目前代码真正没做开的部分

这一点尤其重要。

目前：

```python
if scenario == "q1":
    ...
elif core_of.get(pred) != core:
    ...
```

意味着：

$$
Q2\equiv Q3
$$

在 dependency model 上没有区别。

所以现在所谓 `scenario="q3"`，实际上还没有形成第三问专属算法。

从你们现有代码结构来看，Q3 已经预留出来的最明显信息是：

$$
partition.pipe\_cycles
$$

即：

$$
C_i^M,\quad
C_i^V,\quad
C_i^{MTE},\ldots
$$

partition 中已经累计不同 Pipe 的 cycles。

并且调度时已经计算：

$$
\text{max pipe load}
$$

以及：

$$
I_k=
\frac{
\max_p L_{kp}
}{
\operatorname{avg}_pL_{kp}
}.
$$



所以从**当前代码的设计意图**推断，Q3 最自然的区别应该落在：

> 从 Core-level task scheduling，继续提升到 Pipe/resource-aware scheduling。

但这里我要区分一下：这是根据现有实现结构得到的算法设计推断；你这轮提供的三个代码文件本身没有完整列出 Q3 的原题条件，因此不能仅凭代码断言“题目规定 Q3 就一定是这个模型”。

---

# 十、如果 Q3 允许/强调 Pipe 间流水，那么 partition 策略会再次改变

这时候就非常有意思了。

Q1 里通常倾向：

$$
\text{融合}
$$

因为消除通信。

Q2 中倾向：

$$
\text{适度细分}
$$

因为同 Core partition 间可以复用数据。

到了 Q3，如果不同 Pipe 可以形成流水：

$$
MTE\rightarrow Cube\rightarrow Vector
$$

那么某些不同 Pipe 算子反而应该形成一个**pipeline stage block**。

例如：

$$
COPY/MTE
\rightarrow MATMUL
\rightarrow ADD/RELU
$$

如果三者分别对应：

$$
PIPE_{MTE},
PIPE_M,
PIPE_V,
$$

理想 partition 的执行时间不一定是

$$
C_{MTE}+C_M+C_V,
$$

而可能更加接近资源瓶颈：

$$
C_{stage}
\approx
\max(C_{MTE},C_M,C_V)
$$

——当然具体能否达到这个程度取决于正式 evaluator 的执行规则。

因此 Q3 分块时应该额外奖励：

$$
\text{resource complementarity}.
$$

你们 `_merge_score()` 已经有一个非常初步的版本：

```text
resource_diversity = +0.8
```

当两个 block 使用不同 Pipe 时给予一定收益。

Q3 可以把它正式提升为一个核心项。

---

# 十一、Q3 的 merge score 可以变成

假设 partition \(B\) 的各 Pipe 工作量为：

$$
\mathbf C_B=
(C_B^M,C_B^V,C_B^{MTE}).
$$

定义资源互补收益：

$$
R(A,B)
=
\frac{
\sum_p(C_A^p+C_B^p)
}{
\max_p(C_A^p+C_B^p)
}.
$$

如果两个任务都只压 Cube：

$$
R\approx1.
$$

如果一个主要压 Cube，一个主要压 Vector：

$$
R>1.
$$

说明更有潜力形成资源流水。

于是：

$$
M_{Q3}(A,B)
=
S
+
C_{\rm saved}
+
\eta R(A,B)
-
\mu P_{\rm loss}
-
\nu BufferPressure.
$$

这里还有一个 Q3 很可能必须注意的量：

$$
BufferPressure.
$$

因为流水越激进：

$$
\text{live tensor 数量}\uparrow
$$

很可能同时增加片上 buffer 压力。

你们的 `SemanticBlock` 事实上已经预留了：

```python
buffer_pressure
pipelineability
```

但目前基本没有真正参与算法。

所以这两个变量其实非常适合成为 Q3 的专属特征。

---

# 十二、这样三问的 partition 粒度会形成很有意思的变化

不应该简单认为：

$$
Q1\rightarrow Q2\rightarrow Q3
$$

partition 越来越大。

真正更可能是：

### Q1

为了减少边界代价：

$$
\boxed{\text{通信驱动的大块化}}
$$

### Q2

因为同 Core 可以复用：

$$
\boxed{\text{保留更多调度自由度}}
$$

partition 反而可以适当变小。

### Q3

对于：

$$
MTE\rightarrow Cube\rightarrow Vector
$$

这种硬件互补 motif：

$$
\boxed{\text{跨 Pipe 的 stage 再融合}}
$$

但对于：

$$
Cube\rightarrow Cube
$$

不一定应该合并。

所以 Q3 不再是简单的“大/小 partition”，而是：

$$
\boxed{\text{按硬件流水关系选择性融合}}
$$

这会比单纯调整 `max_ops=16/24/32` 更有理论意义。

---

# 十三、图模式识别本身也应该随三问改变用途

当前 `graph_patterns.py` 用：

* depth；
* width；
* branch ratio；
* join ratio；
* operator ratios；

判断图属于 Wide、Narrow、Mixed、Complex。

这部分不用为三问重写。

真正变化的是：

$$
\text{Pattern}
\times
\text{Question}
\rightarrow
\text{Strategy}.
$$

而现在代码主要是：

$$
\text{Pattern}
\rightarrow
\text{Strategy}.
$$

我建议升级成一个二维策略表：

| 图结构         | Q1                    | Q2                       | Q3                            |
| ----------- | --------------------- | ------------------------ | ----------------------------- |
| Wide        | 大块融合 + balance        | 较细 partition + spreading | Pipe-balanced spreading       |
| Narrow Deep | 长链融合                  | sticky same-core         | stage pipeline                |
| Mixed       | semantic merge        | affinity list            | resource-aware list           |
| Complex     | critical-chain fusion | critical-path sticky     | critical-path + Pipe affinity |

这会让“图识别”真正参与三问，而不是只在当前几个 case 上调参数。

---

# 十四、Wide 图在三问中的变化尤其明显

宽图：

$$
W\gg \#Core.
$$

本来就不缺并行性。

所以 Q1：

$$
\text{可以更大胆融合}
$$

因为即使把 32 个 operator 合成一个 partition：

$$
\text{ready partition 数仍然很多}.
$$

因此主要减少 communication。

Q2 则不需要那么强融合。

因为你完全可以：

$$
P_1,P_2,P_3
$$

留成三个 partition，然后把依赖相关的任务放在相同 Core。

这样 scheduler 的自由度更高。

Q3 如果 Pipe 结构丰富，就应该进一步考虑：

$$
\text{每个 Core 的 Pipe mixture}.
$$

---

# 十五、Narrow Deep 图正相反

假设：

$$
W=2,\qquad D=500.
$$

这时候本来就没有多少 Core-level parallelism。

所以 Q1：

$$
\text{尽量合并长链}
$$

非常合理。

Q2：

即便 partition 切得稍细，只要保持：

$$
P_i@Core_k
\rightarrow
P_{i+1}@Core_k
$$

也没有太大损失。

所以重点从：

$$
partition fusion
$$

转为：

$$
Core affinity.
$$

Q3：

如果链上恰好呈现：

$$
MTE\rightarrow M\rightarrow V
\rightarrow MTE\rightarrow M\rightarrow V
$$

那它反而最适合做：

$$
\boxed{\text{stage pipeline}}
$$

也就是说“窄深图”未必就是 Q3 最差的图。

---

# 十六、Complex 图的变化也很关键

Attention / residual 的主要特点是：

$$
fan\text{-}in,\quad fan\text{-}out
$$

很多。

例如：

$$
A\rightarrow C,
\qquad
B\rightarrow C.
$$

这类 join 上，partition 决策尤其敏感。

Q1 下：

如果把：

$$
A+C
$$

合并，可能消掉 \(A\to C\) 边，但：

$$
B\to C
$$

仍然存在，而且可能破坏并行度。

所以 fan-in 处应该非常保守。

你们现有 semantic merger已经明确拒绝很多 fan-in/fan-out merge，并给它们较强结构惩罚。

Q2 则可以利用：

$$
\text{heavy predecessor affinity}.
$$

即：

$$
core(C)
=
core\left(
\arg\max_{j\in pred(C)}s_{jC}
\right).
$$

其实 `_preferred_complex_core()` 现在就是这个思想：优先选择通信量最大的已调度前驱。

这个策略应该明确归到 Q2，而不是所有 question 都用同一权重。

---

# 十七、因此代码架构最好这样改

现在：

```text
semantic_partition(features)

        ↓

schedule(partitions, scenario)
```

建议改成：

```text
policy = QuestionPolicy(Q1/Q2/Q3)

        ↓

semantic_partition(
    features,
    merge_policy=policy.partition_policy
)

        ↓

schedule(
    partitions,
    cost_model=policy.schedule_policy
)
```

也就是把 question 条件统一抽象成一个：

$$
\boxed{\text{Cost Model}}
$$

例如：

```python
QuestionPolicy:
    same_core_wait
    cross_core_wait
    bandwidth

    communication_weight
    parallelism_weight

    same_core_reuse
    pipe_overlap
    buffer_penalty
```

这样 Q1/Q2/Q3 不需要维护三套算法。

而是：

$$
\boxed{
\text{统一算法框架}
+
\text{不同物理代价模型}
}
$$

这个在数学建模论文里会漂亮很多。

---

# 十八、具体参数层面可以先这样设计

不是说下面的数字必须固定，而是算法方向。

| 参数                       | Q1 |    Q2 |              Q3 |
| ------------------------ | -: | ----: | --------------: |
| partition 通信收益权重         |  高 |     中 |               中 |
| parallelism 保留权重         |  中 |     高 |               高 |
| same-core affinity       |  高 |    很高 |              很高 |
| critical-path 权重         |  高 |     高 |               高 |
| Pipe balance             | 次要 |     中 |              很高 |
| resource complementarity |  低 |     低 |              很高 |
| buffer pressure          |  低 |     中 |               高 |
| singleton repair         | 保守 | 可适当增强 | motif-dependent |

可以把它理解成：

$$
Q1:\quad communication\ boundary
$$

$$
Q2:\quad data\ locality
$$

$$
Q3:\quad data\ locality+resource\ overlap
$$

---

# 十九、一个非常重要的结论

现在不要急着为 Q1/Q2/Q3 各写一套 partitioner。

更合理的是：

$$
\boxed{
M_q(A,B)
=
w_sS
+w_c^{(q)}C
+w_r^{(q)}R
-w_p^{(q)}P
-w_b^{(q)}B
}
$$

其中：

* \(S\)：语义亲和；
* \(C\)：通信收益；
* \(R\)：Pipe / resource complementarity；
* \(P\)：并行度损失；
* \(B\)：buffer / boundary pressure。

三问只是使用不同的：

$$
w^{(q)}.
$$

然后 scheduler 的 Core 选择：

$$
k^*
=
\arg\min_k
\left[
EFT_q(i,k)
+
\lambda_q C_{cross}(i,k)
+
\mu_q I_{pipe}(i,k)
\right].
$$

于是形成一个完整统一模型：

$$
\boxed{
\text{Question-aware Semantic Partitioning}
+
\text{Question-aware List Scheduling}
}
$$

---

## 最值得优先改的三件事

如果现在开始改代码，我会按这个优先级做：

1. **让 `semantic_partition()` 接收 scenario/cost model。**
   Q1 提高消除 partition boundary 的收益，Q2 降低盲目融合倾向，真正让三问产生不同 partition。

2. **Q2 显式加入 Tensor/Core affinity。**
   现在这种效果只是通过 `dependency_ready` 间接产生，最好明确算出

   $$
   CrossBytes(i,k)
   $$

   并进入 placement score。

3. **把 Q3 从 Q2 中真正拆出来。**
   当前代码 Q2/Q3 基本相同。利用已有 `pipe_cycles`、`pipelineability` 和 `buffer_pressure`，让 Q3 的 merge score 和 core score 真正考虑 Pipe 资源结构。当前 partitioner 本身已经预留了 `pipelineability` 和 `buffer_pressure`，只是还没真正用起来。

这样以后论文就不是“我们针对三个问题调了几组经验参数”，而是一个更完整的逻辑：

$$
\boxed{
题设改变物理代价
\Rightarrow
代价函数改变
\Rightarrow
最优分块粒度改变
\Rightarrow
调度策略随之改变
}
$$

这条因果链应该成为 Q1–Q3 算法设计的主轴。
