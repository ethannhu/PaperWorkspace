> 文档定位：本文保留统一数学模型和论文叙事；三道题的规则、Task 语义和算法输出边界以 [`demo_algorithm.md`](demo_algorithm.md) 为准。
>
> 重要区分：Q1 惩罚的是跨**子图**边，即使同核也必须经 DDR；Q2/Q3 惩罚的是跨**核**边。buffer、执行时刻和 Cache 命中由官方评估器决定，本文公式是代理模型。

是的，三问非常适合写成“同一基础模型，随硬件能力逐步增强”的迭代框架，而且这样论文会比三问各做一套算法更完整。

我会把它概括成：

$$
\boxed{
\text{Q1：跨子图都走 DDR}
\;\longrightarrow\;
\text{Q2：同核可驻留复用}
\;\longrightarrow\;
\text{Q3：进一步引入共享 L2}
}
$$

题目本身就是这个递进关系：问题 1 对应场景 A，每个子图就是一个独立 Task，所有跨子图数据都必须经 DDR；问题 2 对应场景 B，同一核心的所有子图合并为一个 Task，同核数据可驻留在 L1/UB，只有跨核依赖需要经过 DDR；问题 3 沿用问题 2 的 Task 结构，再加入容量 1 MB、带宽 250 bytes/cycle 的共享只读 L2 Cache。  

因此，建议不要把三问写成：

$$
Model_1,\quad Model_2,\quad Model_3
$$

而写成统一模型

$$
\mathcal M(\mathcal H)
$$

其中 \(\mathcal H\) 表示硬件通信/存储能力。三问只是逐渐放开约束。

---

### 第一问：先解决“切多大、放哪核、什么顺序”

第一问其实是最纯粹的 partition + scheduling。

因为场景 A 里，只要切开子图，不管两个子图最后是不是放在同一个 Core 上，它们之间的数据都不能留在 L1/UB，而必须重新经过 DDR。评估器甚至规定，每个子图单独构成一个 Task。 

因此这时的 partition 边成本可以非常直接地写成：

$$
C_{ij}^{(1)}
=
d_{ij}\cdot I[p(i)\neq p(j)].
$$

注意这里甚至不用先讨论“跨核还是同核”。

因为：

$$
\boxed{\text{跨 partition}\Rightarrow\text{DDR}}
$$

所以 Q1 里最主要的 trade-off 是：

$$
\text{切得细}
\Rightarrow
\begin{cases}
\text{多核并行度提高}\\
\text{DDR 搬运增加}\\
\text{Task 数增加}
\end{cases}
$$

而：

$$
\text{切得粗}
\Rightarrow
\begin{cases}
\text{局部复用更好}\\
\text{DDR 边界减少}\\
\text{但并行度下降}\\
\text{大子图内部 cache 压力可能上升}
\end{cases}
$$

题目也明确指出，大子图可能因为核内启发式调度和 L1/UB 压力导致额外换入换出，所以“越大越好”并不成立。

因此第一问可以先建立一个基础代价：

$$
J_1
=
\alpha T_{\max}
+\beta D_{\text{cut}}
+\gamma P_{\text{imbalance}},
$$

其中

$$
D_{\text{cut}}
=
\sum_{(i,j)\in E}
d_{ij}I[p(i)\neq p(j)].
$$

这里 \(P_{\text{imbalance}}\) 表示多核负载不均衡。

partition affinity 可以先定义成：

$$
A_{ij}^{(1)}
=
\lambda_f F_{ij}
+\lambda_p P_{ij}
+\lambda_l L_{ij}
-\lambda_d d_{ij}.
$$

也就是：

> 算子适合融合、适合流水、数据局部性强，就尽量不切；但如果保留在一起严重损失多核并行度，就应该切。

第一问里，我甚至会把你前面说的“算子角色识别”主要服务于这一层。

例如：

$$
\text{Cube}\rightarrow\text{Vector}
$$

如果两者有明显 pipeline overlap，就倾向放在同一子图；

但：

```text
        A
       / \
   MatMul MatMul
```

这种宽分支则优先拆开给多个 Core。

---

第一问调度层面可以直接用 partition DAG：

$$
G_P=(P,E_P)
$$

定义每个 partition 的估计执行时间

$$
T_p.
$$

然后 critical rank：

$$
rank_u(p)
=
T_p+
\max_{q\in succ(p)}
\left(C_{pq}+rank_u(q)\right).
$$

再用：

$$
EFT(p,c)=EST(p,c)+T_p
$$

做类似 HEFT 的核分配。

所以第一问实际上就是：

$$
\boxed{
\text{Semantic Partition}
+
\text{Cut Cost}
+
\text{Critical-path Scheduling}
}
$$

这是整个算法的 base model。

---

### 第二问：模型真正发生质变的是“同核通信”

问题 2 不是简单换几个参数。

它引入了一个很重要的新决策：

$$
\boxed{
\text{一个 partition 切开以后，放在同一个 Core 还是不同 Core？}
}
$$

因为场景 B 中，同一核心上的所有子图会合并为一个 Task；同核依赖可以直接保留 L1/UB 中的数据，而只有跨核边才插入 COPY_OUT/COPY_IN。 

所以通信成本从 Q1 的二元模型：

$$
C_{ij}^{(1)}
=
\begin{cases}
0,&same\ partition\\
D_{ij},&different\ partition
\end{cases}
$$

变成三级：

$$
C_{ij}^{(2)}
=
\begin{cases}
0,
&same\ partition
\\[1mm]
C_{ij}^{local},
&different\ partition,\ same\ core
\\[1mm]
C_{ij}^{DDR},
&different\ core
\end{cases}
$$

而且

$$
C_{ij}^{local}\ll C_{ij}^{DDR}.
$$

这时 partition 和 core assignment 就不能完全割裂了。

第一问可以：

$$
\text{先切图}\rightarrow\text{再分核}
$$

但第二问最好开始做：

$$
\boxed{
\text{partition-core co-optimization}
}
$$

即切图时已经考虑“这些 partition 将来有没有可能驻留在同一核”。

---

更关键的是，Q2 引入了 buffer lifetime。

假设 tensor \(x\) 是 \(P_a\) 产生、\(P_b\) 消费，并且两者在同一个 Core：

$$
core(a)=core(b).
$$

那么 tensor 可以不写回 DDR，而保持驻留：

$$
x\in L1/UB.
$$

其生命周期大概是：

$$
life(x)
=
[F_a^{produce},S_b^{consume}].
$$

因此任一 Core \(c\) 的缓存约束变成：

$$
\sum_{x:\tau\in life(x),\,core(x)=c}
size(x)
+
M_{\text{working}}(\tau)
\le M_c.
$$

这就是 Q2 相比 Q1 最核心新增的状态。

题目也明确说，同核数据驻留会持续占用私有缓存，可能压缩中间子图的可用空间，从而降低执行效率。

所以 Q2 目标函数应升级为：

$$
J_2
=
\alpha T_{\max}
+
\beta D_{\text{cross-core}}
+
\gamma M_{\text{pressure}}
+
\eta Spill.
$$

其中：

$$
D_{\text{cross-core}}
=
\sum_{(i,j)}
d_{ij}
I[core(i)\neq core(j)].
$$

注意此时已经不是 penalize “cut”。

而是 penalize：

$$
\boxed{\text{cross-core cut}}
$$

这点非常重要。

因为：

```text
P1 → P2
```

在 Q1 中，只要 \(P_1,P_2\) 是两个 partition，就产生 DDR。

但 Q2：

```text
Core 0:
P1 → P2
```

完全可以核内复用。

所以第二问的聚类目标实际上从：

$$
\text{operator}\rightarrow partition
$$

扩展为：

$$
\text{operator}
\rightarrow
partition
\rightarrow
core-cluster.
$$

你可以把一个 Core 上全部 partition 看作一个更高层的“super partition”。

---

甚至可以定义两层 affinity。

算子层：

$$
A_{ij}^{op}
$$

决定是否属于同一个 partition。

partition 层：

$$
A_{pq}^{core}
=
\alpha D_{pq}
+\beta Reuse_{pq}
-\gamma ParallelLoss_{pq}
-\eta MemoryPressure_{pq}.
$$

如果

$$
A_{pq}^{core}
$$

很高，就把 \(P_p,P_q\) 放到同一 Core。

这个结构我觉得很适合论文，因为会变成漂亮的 hierarchical clustering：

$$
\boxed{
Operator
\rightarrow
Subgraph
\rightarrow
Core Task
}
$$

---

调度部分也相应变化。

Q1 中 partition 是 Task：

$$
Task=P.
$$

Q2 中：

$$
Task_c=\bigcup_{p:core(p)=c}P_p.
$$

也就是说：

$$
\boxed{
\text{Q1 调度对象 = partition}
}
$$

而

$$
\boxed{
\text{Q2 全局同步对象 = Core Task}
}
$$

但 Core 内部仍保留 partition 顺序。

所以 Q2 的调度优化会变成双层：

$$
\text{global inter-core schedule}
+
\text{local intra-core partition order}.
$$

此时 critical path 还可以继续沿用，只需要把通信项改掉：

$$
rank_u^{(2)}(p)
=
T_p+
\max_q
\left[
C_{pq}^{(2)}+rank_u^{(2)}(q)
\right].
$$

其中如果计划把 \(p,q\) 放同一 Core：

$$
C_{pq}^{(2)}\approx 0
$$

或者仅用缓存驻留代价。

---

### 第三问：再引入“共享输入是否值得进 L2”

第三问沿用第二问的 Task 结构，只增加共享只读 Cache。评估器也是明确这样做的。 

这意味着：

$$
Partition
$$

基本模型可以不变，

$$
Core Assignment
$$

基本模型也不变，

真正变化的是 memory traffic model。

Q2 时，一个共享输入 tensor \(x\) 如果被多个 Core 使用：

```text
          x in DDR
        /    |    \
     Core0 Core1 Core2
```

那么三个核通常都从 DDR 读取：

$$
D_{DDR}(x)
\approx
3size(x).
$$

Q3 引入共享只读 L2 后：

```text
              DDR
               |
               x
               |
              L2
          /    |    \
       C0     C1     C2
```

后续读可能由 L2 提供。

因此输入 \(x\) 的有效通信成本可以改成：

$$
C_x^{(3)}
=
C_{miss}(x)+C_{hit}(x).
$$

例如粗略写：

$$
C_x^{(3)}
=
\frac{size(x)}{B_{DDR}}
+
(n_x-1)\frac{size(x)}{B_{L2}}
$$

而 Q2 是：

$$
C_x^{(2)}
=
n_x\frac{size(x)}{B_{DDR}}.
$$

其中 \(n_x\) 是访问该共享输入的 Core 数。

于是 L2 带来的理论收益：

$$
Gain_{L2}(x)
=
C_x^{(2)}-C_x^{(3)}.
$$

展开：

$$
Gain_{L2}(x)
\approx
(n_x-1)size(x)
\left(
\frac1{B_{DDR}}
-
\frac1{B_{L2}}
\right).
$$

所以共享度越高：

$$
n_x\uparrow
$$

L2 越有价值。

---

因此 Q3 的算子/数据特征里，应该新增一个非常重要的 feature：

$$
fanout(x)
$$

以及：

$$
reuseScore(x)
=
fanout(x)\cdot size(x).
$$

甚至进一步定义：

$$
L2Score(x)
=
fanout(x)
\cdot size(x)
\cdot criticality(x).
$$

也就是说：

> 同一个大输入被很多 Core 反复读取，而且位于关键路径附近，那么它就是最值得利用 L2 的数据。

这时你前面提到的“高维图特征”就真正发挥作用了。

Q3 不仅看算子：

$$
op\ type,\ pipe,\ cycles
$$

还需要看 tensor：

$$
size,\ fanout,\ reuse,\ lifetime,\ core\ spread.
$$

---

不过 L2 不是无限免费的。

容量只有：

$$
M_{L2}=1\text{ MB},
$$

带宽为：

$$
B_{L2}=250\text{ bytes/cycle},
$$

而题目的 DDR 总带宽是固定共享资源。 

所以必须考虑：

$$
\sum_{x\in Cache}size(x)
\le M_{L2}.
$$

如果抽象成显式 cache placement，可以定义：

$$
h_x\in\{0,1\}
$$

表示 \(x\) 是否希望作为高复用数据保留在 L2。

目标就是：

$$
\max
\sum_x
h_x Gain_{L2}(x)
$$

满足：

$$
\sum_xh_xsize(x)
\le M_{L2}.
$$

这其实近似一个 knapsack。

当然实际评估器是 Cache，不是让我们直接指定“把哪个 tensor 放进去”，因此算法输出不能直接控制 \(h_x\)。

但是我们可以通过切图和 core assignment 间接影响：

$$
\boxed{
\text{哪些输入会被多个 Core 重复访问}
}
$$

也就是做 Cache-aware partitioning。

---

比如一个输入 \(X\) 被四个彼此独立的计算分支使用：

```text
             X
       /     |     |     \
      A      B     C      D
```

Q2 中你可能觉得：

> 重复 DDR 读取太贵，尽量把 A/B/C/D 聚到少数 Core。

因为：

$$
D_{DDR}\uparrow
$$

但 Q3 中因为 L2 可以服务共享输入，你可以更大胆地：

$$
A,B,C,D
\rightarrow
4\ cores
$$

换取更多并行度。

因此 L2 实际改变了 partition 的“最优粒度”。

这点非常值得写进论文：

$$
\boxed{
L2降低跨核共享输入的边际通信成本，
因此最优策略会从 locality-oriented
向 parallelism-oriented 偏移。
}
$$

也就是说，三个问题并不只是硬件越来越强，而是 Pareto 最优点发生迁移：

```text
Q1:
communication <<<<<< parallelism

Q2:
locality + memory residence ↔ parallelism

Q3:
shared-read cost下降
          ↓
可以进一步增加跨核并行
```

---

所以我建议最后把统一模型写成下面这样。

定义：

$$
x_{ip}
$$

——节点 \(i\) 属于 partition \(p\)。

$$
z_{pc}
$$

——partition \(p\) 属于 Core \(c\)。

统一优化目标：

$$
\boxed{
J=
\alpha T_{\max}
+\beta D_{DDR}
+\gamma S_{spill}
+\eta M_{pressure}
-\mu G_{L2}
}
$$

然后三个问题通过关闭/打开不同项得到。

问题 1：

$$
\boxed{
J_1
=
\alpha T_{\max}
+\beta D_{\text{cut}}
}
$$

且：

$$
p_i\neq p_j
\Rightarrow DDR.
$$

问题 2：

$$
\boxed{
J_2
=
\alpha T_{\max}
+\beta D_{\text{cross-core}}
+\gamma S_{spill}
+\eta M_{residence}
}
$$

且：

$$
core_i=core_j
\Rightarrow
\text{允许 L1/UB resident reuse}.
$$

问题 3：

$$
\boxed{
J_3
=
\alpha T_{\max}
+\beta D_{DDR}
+\gamma S_{spill}
+\eta M_{residence}
-\mu G_{L2}.
}
$$

其中

$$
G_{L2}
=
\sum_x
Reuse_x^{cross-core}
\cdot HitBenefit_x.
$$

这样就形成了非常清晰的递进：

$$
\boxed{
\begin{array}{c}
Q1\\
\text{partition-aware}
\end{array}
\quad\rightarrow\quad
\boxed{
\begin{array}{c}
Q2\\
\text{core-locality-aware}
\end{array}
\quad\rightarrow\quad
\boxed{
\begin{array}{c}
Q3\\
\text{shared-cache-aware}
\end{array}
}
$$

算法也最好不要换。

可以始终保持同一个框架：

$$
\text{Feature Extraction}
\rightarrow
\text{Initial Partition}
\rightarrow
\text{Core Assignment}
\rightarrow
\text{Schedule}
\rightarrow
\text{Local Improvement}.
$$

只是评分函数逐问升级。

Q1 的 move gain：

$$
\Delta J_1
=
\Delta T+
\lambda\Delta D_{\text{cut}}.
$$

Q2：

$$
\Delta J_2
=
\Delta T+
\lambda_1\Delta D_{\text{cross-core}}
+
\lambda_2\Delta MemoryPressure.
$$

Q3：

$$
\Delta J_3
=
\Delta T+
\lambda_1\Delta D_{DDR}
+
\lambda_2\Delta MemoryPressure
-\lambda_3\Delta L2Reuse.
$$

这样你甚至可以写一个统一的局部搜索器：

```text
initial partition
        ↓
initial core assignment
        ↓
critical-path schedule
        ↓
repeat
    merge partitions
    split partition
    move partition to another core
    swap partition order
    exchange partitions between cores

    evaluate ΔJ(scene)
    accept profitable move
until convergence
```

而真正变化的只有：

```text
score(scene=A)
score(scene=B)
score(scene=B+L2)
```

这比给三问各写一套算法要漂亮很多。

如果往最后的论文结构压缩，我会直接把三问命名成：

$$
\boxed{
\begin{aligned}
\text{模型 I：}&\ \text{DDR 边界感知的多核图划分调度}\\
\text{模型 II：}&\ \text{缓存驻留感知的分层图划分调度}\\
\text{模型 III：}&\ \text{共享 L2 复用感知的多核调度}
\end{aligned}
}
$$

而底层算法统一叫类似：

> **语义—通信—存储联合感知的层次化 DAG 划分与关键路径调度算法**

这个叙事会把三个问题很好地串成一条线：不是“第一问做 HEFT，第二问另想办法，第三问再补 Cache”，而是**同一个模型随着硬件从 DDR-only → local-resident → shared-cache 逐层解除通信约束，优化重点也从减少 cut，转向核内驻留，再转向跨核共享复用与并行度释放。**
