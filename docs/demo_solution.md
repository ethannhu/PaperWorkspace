26 年 A 题我建议把它理解成一个**“面向 NPU 的计算图联合划分与多核调度”问题**。和 25 年相比，26 年已经不是重点研究单核内部怎么排操作，而是把官方核内调度器当成底层执行器，你负责决定：

**怎么切图 → 每块放哪个核 → 各核按什么顺序执行。**

最终目标是 Makespan 最小，同时尽量降低额外 DDR 搬运。官方评估器已经会处理 L1/UB 换入换出、Pipe 并行以及 DDR 带宽竞争，所以没必要自己重新实现一个完整硬件模拟器。

我会采用下面这条总体路线：

> **通信感知图聚类 → 关键路径感知多核分配 → 缓存感知核内排序 → 官方评估器驱动的局部搜索**

这基本可以贯穿问题 1～3。

---

### 1. 先把整个问题抽象清楚

原计算图是 DAG，可以把非 COPY 操作作为待分配节点。

对于每个计算节点 \(v\)，提取几个特征：

$$
v=(t_v,\ pipe_v,\ L1_v,\ UB_v)
$$

其中 \(t_v\) 是计算周期，`pipe` 是 Cube/Vector 等。

对于每条依赖边 \(e=(u,v)\)，记录对应张量大小：

$$
w_e=\text{tensor size}
$$

于是本质上是一个带权 DAG。

我们最终需要求三个东西：

$$
\boxed{
\text{节点}\rightarrow\text{子图}
\rightarrow\text{核心}
\rightarrow\text{核内顺序}
}
$$

实际目标可以写成：

$$
\min T_{\mathrm{makespan}}
$$

同时把

$$
M_{\mathrm{DDR}}
$$

作为次目标。

但我不建议真的试图推导一个精确的 Makespan 解析公式，因为真实时间还受 Pipe 重叠、缓存换入换出、DDR 竞争影响。官方模拟器本身就是最准确的目标函数。

所以更适合：

$$
\text{启发式近似代价}
\rightarrow
\text{筛选候选方案}
\rightarrow
\text{官方 evaluator 精确评分}
$$

---

## 2. 问题 1：先解决“并行度 vs DDR 通信”的矛盾

问题 1 的特点非常关键：

**一个子图就是一个 Task。**

而且即使两个连续子图放在同一个核，它们之间的数据也不能直接保留在缓存里，仍然要经过 DDR。

同时：

* 同核 Task 切换等待：100 cycles；
* 跨核前驱同步：1000 cycles；
* DDR 总带宽：60 bytes/cycle。

所以问题 1 最忌讳的做法就是：

> 一个 op 切成一个子图。

理论并行度很高，但通信和 Task 开销直接爆炸。

### 我的第一步会做 Heavy-edge Coarsening

定义两个相邻节点合并的收益：

$$
G(u,v)
=
\alpha w_{uv}
+\beta R(u,v)
-\gamma P(u,v)
$$

其中：

* \(w_{uv}\)：二者之间的数据量；
* \(R(u,v)\)：共享输入带来的数据复用收益；
* \(P(u,v)\)：合并以后损失的并行度。

如果 \(u,v\) 之间传的是一个很大的 tensor，那么应该强烈倾向于合并。

这其实就是**通信感知图聚类**。

题目本身也明确指出，大量共享数据的节点应尽量聚在一起，否则会造成重复 DDR 读取。

---

### 然后做 HEFT 风格调度

得到几十个或者几百个“粗粒度子图”后，再分配给 \(K=2\sim5\) 个核。

这里非常适合 HEFT（Heterogeneous Earliest Finish Time）的思路。

先计算每个子图的 upward rank：

$$
rank_u(i)
=
t_i+
\max_{j\in succ(i)}
(c_{ij}+rank_u(j))
$$

其中通信代价近似为：

$$
c_{ij}
\approx
\frac{2\,D_{ij}}{60}
+
T_{\mathrm{sync}}
$$

然后按照 `rank` 从大到小做列表调度：

对每个子图 \(i\)，尝试放到每个核心 \(k\)，估算：

$$
EFT(i,k)
$$

选择最早完成的核心。

这样天然就会优先保护关键路径。

---

## 3. 但仅仅 HEFT 还不够，问题 1 真正应该做的是“切图 + 调度联合优化”

比如出现：

```text
        B
       /
A ----
       \
        C
```

如果 A→B、A→C 都传输大量数据，把三个全部放一个核通信最小，但没有并行。

如果把 B、C 分开：

```text
core0: A -> B

core1:      C
```

又产生重复通信。

因此真正优化的是：

$$
\boxed{
并行收益
-
通信代价
-
同步代价
}
$$

所以在得到一个 HEFT 初始解之后，我会做局部搜索。

只需要设计四五种邻域操作：

1. 合并两个相邻子图；
2. 拆分一个大子图；
3. 把子图迁移到另一个核；
4. 交换两个核的子图；
5. 调整同一核内两个无依赖子图的执行顺序。

每次不需要遍历所有方案。

例如当前瓶颈核：

```text
core0 ████████████████
core1 █████████
core2 ████████
core3 █████████
```

只围绕 core0 上的子图尝试迁移和拆分。

这就是典型的**关键路径局部搜索**。

---

# 4. 问题 2 的思路要明显变化

问题 2 表面上只是换成场景 B，但实际上优化结构变化非常大。

因为：

> 同一个核心上的所有子图最终会被合成一个 Task。

同核依赖不用经过 DDR，可以直接使用 L1/UB 中的数据；只有跨核依赖才插入 COPY。

跨核 COPY 同步延迟变成 500 cycles。

所以问题 2 的第一层决策其实可以简化成：

$$
\boxed{
计算图 \rightarrow K个核心分区
}
$$

而不需要像问题 1 那样特别纠结大量小 Task。

---

### 问题 2 可以建成“通信感知负载均衡图划分”

设节点 \(v\) 分配到核 \(k\)：

$$
x_{vk}\in\{0,1\}
$$

考虑三个代价。

第一项是负载：

$$
C_\text{load}
=
\max_k
\sum_v t_vx_{vk}
$$

第二项是跨核通信：

$$
C_\text{cut}
=
\sum_{(u,v)\in E}
w_{uv}
[x_u\ne x_v]
$$

第三项非常重要，是缓存压力：

$$
C_\text{cache}
$$

于是可以做一个代理模型：

$$
J=
\alpha C_\text{load}
+\beta C_\text{cut}
+\gamma C_\text{cache}
$$

前三项都不是最终评分，而是帮我们快速找候选方案。

真正好坏还是交给 evaluator。

---

# 5. 问题 2 最值得做创新的是 Cache-aware Scheduling

这一点我觉得很可能是论文里比较好写的部分。

官方核内调度会：

* 维护 tensor 驻留；
* 空间不够时选择 tensor 换出；
* 未来再次使用时 COPY_IN；
* 采用类似“下一次使用最晚”的策略选择换出对象。

因此如果你的子图顺序不好：

```text
produce X
A
B
C
D
E
use X
```

那么 X 会在缓存中活很久。

其 live range 很长：

```text
X: ├────────────────────────────┤
```

非常容易造成缓存压力。

如果换成：

```text
produce X
use X
A
B
C
D
E
```

则：

```text
X: ├───┤
```

缓存占用骤降。

所以核内子图排序可以设计一个优先级：

$$
Priority(i)
=
a\cdot Criticality(i)
+b\cdot ReleaseBytes(i)
+c\cdot Reuse(i)
$$

这里 `ReleaseBytes` 表示执行该子图以后可以释放多少 L1/UB tensor。

于是：

> 不只优先执行“最快完成”的节点，还优先执行“能够尽快释放大 tensor”的节点。

这个很适合称作：

**Memory-pressure-aware list scheduling**

---

# 6. 问题 2 甚至可以显式估算 tensor live range

设 tensor \(x\)：

* producer 位于位置 \(p_x\)
* 最后 consumer 位于位置 \(l_x\)

则近似驻留区间：

$$
[p_x,l_x]
$$

在调度位置 \(t\) 时，缓存需求：

$$
M(t)=
\sum_x
size(x)
I(p_x\le t\le l_x)
$$

于是：

$$
M^{peak}
=
\max_t M(t)
$$

分别计算：

$$
M^{peak}_{L1}
,\qquad
M^{peak}_{UB}
$$

题目的固定容量是：

$$
L1=512KB
$$

$$
UB=128KB
$$

所以一个非常实用的启发式就是：

$$
\min
\left[
\max_tM_{L1}(t)
+
\lambda \max_tM_{UB}(t)
\right]
$$

把它作为局部搜索代价。

官方题目本身也特别提醒：子图过大或者驻留数据过多，可能引发大量换入换出，甚至出现“合并之后反而更慢”。

---

# 7. 问题 3 不应该重新发明一个算法

问题 3 最合理的做法是：

> **继承问题 2 的算法，然后加入 Cache reuse-aware 项。**

因为问题 3 的 Task 结构和问题 2 相同，只多了：

$$
L2=1MB
$$

$$
BW_{L2}=250\ bytes/cycle
$$

而 DDR：

$$
BW_{DDR}=60\ bytes/cycle
$$

所以 L2 命中相当值钱。

---

## 8. 问题 3 的关键是“Reuse Distance”

L2 是 FIFO Cache。

假设输入 X 被：

```text
core0
core1
core2
```

反复读取。

第一次：

```text
DDR -> L2 -> core0
```

随后 core1/core2 再读取，如果 X 还没被 FIFO 淘汰：

```text
L2 -> core1
L2 -> core2
```

那么就可以绕开 60 B/cycle 的 DDR。

因此需要尽量让：

$$
\text{同一 tensor 的多个访问靠得更近}
$$

可以定义 reuse distance：

$$
RD(x)
=
\sum_{\text{两次访问之间}}
size(y)
$$

如果：

$$
RD(x)<1MB
$$

则比较可能仍驻留在 L2。

所以问题 3 可以增加：

$$
C_{L2}
=
\sum_x
size(x)\cdot miss\_estimate(x)
$$

优化目标变成：

$$
J=
\alpha C_\text{load}
+
\beta C_\text{cross-core}
+
\gamma C_\text{spill}
+
\delta C_{L2}
$$

---

# 9. 因而我认为整个 26 年 A 题最适合做成一个四层算法

可以画成论文里的总框架：

```text
             原始计算 DAG
                  │
                  ▼
        ┌──────────────────┐
        │ 图结构特征提取   │
        │ compute / tensor │
        │ pipe / reuse     │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Heavy-edge       │
        │ Graph Coarsening │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Critical-path    │
        │ Core Assignment  │
        │     (HEFT)       │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Cache-aware      │
        │ Local Scheduling │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Local Search     │
        │ merge/split/move │
        └────────┬─────────┘
                 │
                 ▼
          官方 Evaluator
                 │
                 ▼
            Makespan
```

这套框架三个问题都能复用。

区别只是代价函数：

| 问题 | 最重要的矛盾                  |
| -- | ----------------------- |
| Q1 | 并行度 vs 跨子图通信            |
| Q2 | 并行度 vs 跨核通信 vs L1/UB 驻留 |
| Q3 | Q2 + L2 数据复用            |

---

## 10. 实际比赛，我不会一上来做复杂算法

我会按这个顺序实现。

**Baseline 1**

拓扑排序 → 均匀切成 \(K\) 块 → 每核一块。

先保证 JSON 100% 合法。

**Baseline 2**

加入计算量负载均衡：

$$
Load_k\approx\sum t_v
$$

**Baseline 3**

加入 edge weight：

大 tensor 强依赖节点尽量放一起。

这时候实际上已经是：

> communication-aware graph partitioning。

**Baseline 4**

加入 HEFT / critical path。

**Baseline 5**

加入局部搜索：

```text
move
swap
merge
split
reorder
```

**最终版**

Q2 加 cache pressure；

Q3 加 L2 reuse distance。

这个开发路线很重要，因为题目要求算法避免暴力迭代，并希望单个用例大约 5～10 分钟内能给出高质量方案。

---

还有一个我认为相当重要的技巧：

**把官方 evaluator 当成 oracle，而不是仅仅在最后评测一次。**

例如局部搜索：

```python
solution = initial_solution()

while True:
    candidates = generate_neighbors(solution)

    # 用快速代理模型先从1000个候选筛到20个
    candidates = surrogate_top20(candidates)

    # 再调用官方 evaluator
    best = min(candidates, key=official_evaluator)

    if best >= solution:
        break

    solution = best
```

这样既利用了数学模型解释问题，也不会因为你的时间估算模型不准，把真正优秀的方案过滤掉。

如果现在让我直接开始做这题，我会优先实现 **“Heavy-edge 聚类 + HEFT + move/merge 局部搜索”作为 Q1 第一版**。它代码量不算大，而且自然可以一路扩展到 Q2/Q3；之后最值得花精力的创新点，则是 **Q2 的 tensor live-range / cache-pressure-aware 调度**。这条线比直接上遗传算法、模拟退火或整数规划靠谱得多。
