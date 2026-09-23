# 题目数据约定与评估接口

本文整理 `challenge.docx` 附录 B 的数据格式、方案格式和评估流程，供
`src/subgraph` 下的切图算法使用。原始题目文档和评估实现仍以项目中的
`challenge.docx` 与 `src/subgraph/artifacts` 为准。

## 1. 材料包目录

当前材料位于 `src/subgraph/artifacts/`：

```text
artifacts/
├── README.md
├── code/
│   ├── multicore_cut_evaluate_problem_1.py
│   ├── multicore_cut_evaluate_problem_2.py
│   ├── multicore_cut_evaluate_problem_3.py
│   ├── schedule_step1.py
│   ├── schedule_step2.py
│   ├── schedule_step3.py
│   └── stub_multicore_cut_and_schedule.py
├── data/
│   ├── case_001.json ... case_100.json
│   └── config.txt
└── docs/
    ├── 核内调度算法.md
    └── 多核并行模拟执行算法.md
```

目录职责：

- `code/`：问题 1～3 的评估器和核内调度辅助程序；
- `data/`：正式测试图和固定评估配置；
- `docs/`：评估模型和核内调度算法说明；
- `README.md`：材料包使用说明。

评测结果必须使用固定的 `data/config.txt`，不得修改该配置文件。

## 2. 原始计算图 JSON

计算图使用 UTF-8 编码，顶层必须包含三个数组：

```json
{
  "tensors": [],
  "ops": [],
  "edges": []
}
```

所有操作 ID 和张量 ID 共用同一个 ID 空间，在同一张图中必须全局唯一。
所有 ID、张量大小和操作周期数必须是非负整数。

### 2.1 `tensors`

每个张量包含：

```json
{
  "id": 1000000000,
  "pos": "DDR",
  "size": 1152
}
```

字段含义：

- `id`：张量唯一 ID；
- `pos`：逻辑存储位置，只能是 `DDR`、`L1` 或 `UB`；
- `size`：张量大小，单位为 bytes。

输入张量和最终输出张量位于 `DDR`；中间张量位于 `L1` 或 `UB`。

### 2.2 `ops`

每个操作包含：

```json
{
  "id": 10,
  "op": "ADD",
  "pipe": "PIPE_V",
  "cycles": 4
}
```

字段含义：

- `id`：操作唯一 ID；
- `op`：操作名称；
- `pipe`：执行流水线，必须是以下之一：
  - `PIPE_MTE2`
  - `PIPE_MTE3`
  - `PIPE_M`
  - `PIPE_V`
- `cycles`：核内操作的执行周期数。

特殊操作：

- `COPY_IN`：DDR → L1/UB；通常使用 `PIPE_MTE2`；
- `COPY_OUT`：L1/UB → DDR；通常使用 `PIPE_MTE3`。

评估器直接读取 `pipe` 字段，不根据 `op` 名称推断流水线。
COPY 的实际时间由数据量和评估时刻的有效带宽决定，`cycles` 不作为其最终
搬运时间。

### 2.3 `edges`

每条边包含：

```json
{
  "source": 1000000000,
  "target": 10
}
```

合法边只能是：

```text
Tensor → Op
Op     → Tensor
```

边表示输入输出依赖，不单独保存通信量。搬运量由关联张量的 `size` 决定。
不允许直接出现 `Tensor → Tensor` 边，也不允许重复边。

题目提供的原始图保证：

- 每个 DDR 输入只搬入一次；
- 每个最终结果只搬出一次；
- 中间计算不会与 DDR 交换数据；
- 单个计算操作的输入输出总量不超过对应片上缓存容量。

## 3. 多核方案 JSON

参赛算法输出文件名为：

```text
<case>_multicore_res.json
```

其中 `<case>` 是输入 JSON 文件名去掉 `.json` 后的部分。

方案顶层必须且只能包含两个字段：

```json
{
  "node_to_subgraph": {},
  "core_schedules": []
}
```

### 3.1 `node_to_subgraph`

该对象将每个非 COPY 操作映射到一个子图 ID：

```json
{
  "11": 0,
  "12": 1
}
```

约束：

- 键必须是操作 ID 的十进制字符串；
- `COPY_IN` 和 `COPY_OUT` 不应出现在这里；
- 必须恰好覆盖所有非 COPY 操作；
- 值必须是非负整数子图 ID；
- 一个操作只能属于一个子图。

### 3.2 `core_schedules`

这是一个二维整数数组：外层下标就是核心 ID，内层数组是该核心上的子图
执行顺序。

```json
{
  "core_schedules": [
    [0, 2],
    [1],
    []
  ]
}
```

约束：

- 外层数组长度等于使用的核心数；
- 内层数组可以为空；
- 每个子图 ID 必须出现且只能出现一次；
- 同一核心上的顺序不能违反子图依赖；
- 所有子图之间形成的依赖图必须无环。

评估器会自动根据原图、子图归属和场景规则插入跨子图或跨核心的 COPY，算法
不需要手动生成这些 COPY。

## 4. 固定配置

配置文件为 `src/subgraph/artifacts/data/config.txt`：

```ini
[capacity]
L1 524288
UB 131072

[bandwidth]
bandwidth 60

[multicore_scene_a]
task_cross_core_wait_cycles 1000
task_same_core_wait_cycles 100

[multicore_scene_b]
cross_core_copy_delay_cycles 500

[problem_3]
cache_capacity_bytes 1048576
cache_bandwidth_bytes_per_cycle 250
```

参数含义：

- L1：524288 bytes；
- UB：131072 bytes；
- 所有核心共享 DDR 带宽：60 bytes/cycle；
- 问题 1 跨核前驱等待：1000 cycles；
- 问题 1 同核 Task 切换等待：100 cycles；
- 问题 2、3 跨核 COPY 同步延迟：500 cycles；
- 问题 3 只读 Cache：1048576 bytes，带宽 250 bytes/cycle。

评估器默认从计算图所在目录读取 `config.txt`，也可以通过 `--config` 显式
指定。配置缺失或参数非法时评估器直接报错，不使用代码默认值。

## 5. 运行流程

### 5.1 生成方案

自己的算法负责读取计算图并生成标准方案。例如空模板：

```bash
python3 src/subgraph/empty_algorithm.py \
  src/subgraph/artifacts/data/case_001.json \
  -n 4 \
  -o /tmp/case_001_multicore_res.json
```

仅用于验证接口的官方随机示例为：

```bash
python3 src/subgraph/artifacts/code/stub_multicore_cut_and_schedule.py \
  src/subgraph/artifacts/data/case_001.json -n 4
```

该 stub 只生成格式合法的随机方案，不是性能基线。

### 5.2 运行三个问题的评估器

显式指定方案文件：

```bash
python3 src/subgraph/artifacts/code/multicore_cut_evaluate_problem_1.py \
  src/subgraph/artifacts/data/case_001.json \
  /tmp/case_001_multicore_res.json \
  --config src/subgraph/artifacts/data/config.txt

python3 src/subgraph/artifacts/code/multicore_cut_evaluate_problem_2.py \
  src/subgraph/artifacts/data/case_001.json \
  /tmp/case_001_multicore_res.json \
  --config src/subgraph/artifacts/data/config.txt

python3 src/subgraph/artifacts/code/multicore_cut_evaluate_problem_3.py \
  src/subgraph/artifacts/data/case_001.json \
  /tmp/case_001_multicore_res.json \
  --config src/subgraph/artifacts/data/config.txt
```

如果省略第二个位置参数，评估器会在输入图所在目录寻找：

```text
<case>_multicore_res.json
```

评估器不会在方案缺失时自动生成方案。

### 5.3 输出文件

每个问题默认生成三类文件：

```text
<case>_problem_1_res.json
<case>_problem_1_trace.json
<case>_problem_1_log.txt
```

问题编号替换为 `problem_2` 或 `problem_3` 即可。

- `*_res.json`：完整评估结果；
- `*_trace.json`：Perfetto/Chrome Trace Event 格式的时间线；
- `*_log.txt`：便于快速查看 Makespan、搬运量、缓存和时间段的日志。

输出位置可以通过以下参数修改：

```text
-o                 结果 JSON
--trace-output     Trace JSON
--log-output       简短日志
```

## 6. 最小示例

输入图：

```json
{
  "tensors": [
    {"id": 1, "pos": "DDR", "size": 16},
    {"id": 2, "pos": "UB",  "size": 16},
    {"id": 3, "pos": "UB",  "size": 16},
    {"id": 4, "pos": "DDR", "size": 16}
  ],
  "ops": [
    {"id": 10, "op": "COPY_IN",  "pipe": "PIPE_MTE2", "cycles": 1},
    {"id": 11, "op": "ADD",       "pipe": "PIPE_V",   "cycles": 4},
    {"id": 12, "op": "COPY_OUT", "pipe": "PIPE_MTE3", "cycles": 1}
  ],
  "edges": [
    {"source": 1,  "target": 10},
    {"source": 10, "target": 2},
    {"source": 2,  "target": 11},
    {"source": 11, "target": 3},
    {"source": 3,  "target": 12},
    {"source": 12, "target": 4}
  ]
}
```

对应的两核方案只需要描述非 COPY 操作：

```json
{
  "node_to_subgraph": {"11": 0},
  "core_schedules": [[0], []]
}
```

在固定配置下，三个问题的 Makespan 均为 6 cycles，原始及调度后 COPY 总量
均为 32 bytes，新增搬运量为 0。该示例没有可复用输入，因此问题 3 的 Cache
命中次数为 0。
