# 数据约定与评估接口

实际材料位于仓库根目录的 `artifacts/`：

```text
artifacts/data/       原始图和 config.txt
artifacts/excases/    较小的额外样例
artifacts/code/       官方评估器和核内调度代码
artifacts/docs/       官方评估说明
```

## 原始图 JSON

顶层包含 `tensors`、`ops`、`edges` 三个数组。操作 ID 和 Tensor ID 共用全局 ID
空间，且不能重复。

### Tensor

```json
{"id": 100, "pos": "DDR", "size": 1152}
```

`pos` 是 `DDR`、`L1` 或 `UB`，`size` 以 bytes 计。边只能表示 `Op -> Tensor`、
`Tensor -> Op` 或评估器支持的直接 `Op -> Op` 依赖；不应手动加入 Tensor 到 Tensor
的边。

### Op

```json
{"id": 10, "op": "ADD", "pipe": "PIPE_V", "cycles": 4}
```

合法 Pipe 为 `PIPE_MTE2`、`PIPE_MTE3`、`PIPE_M`、`PIPE_V`。`COPY_IN` 和
`COPY_OUT` 是评估器使用的通信操作，算法的 `node_to_subgraph` 不应包含它们。

## 算法方案 JSON

方案只包含两个字段：

```json
{
  "node_to_subgraph": {"11": 0, "12": 1},
  "core_schedules": [[0, 2], [1], []]
}
```

`node_to_subgraph` 必须覆盖所有非 COPY 操作且每个操作只能出现一次。键是操作
ID 的字符串，值是非负子图 ID。

`core_schedules` 的外层下标是 Core ID；每个子图 ID 必须恰好出现一次，同一
Core 上的顺序不能违反依赖。外层长度就是使用的 Core 数。

算法不需要生成 COPY、ALLOC、FREE、SPILL 或核内 Pipe 操作；官方评估器会根据
方案和题目场景生成并检查这些内容。

## 配置

默认从输入图所在目录读取 `config.txt`。正式数据的配置在
`artifacts/data/config.txt`，包括：

- `[capacity]`：`L1`、`UB` 容量；
- `[bandwidth]`：DDR 带宽；
- `[multicore_scene_a]`：Q1 同核、跨核等待；
- `[multicore_scene_b]`：Q2/Q3 跨核 COPY 延迟；
- `[problem_3]`：Q3 Cache 容量和 Cache 带宽。

评估器不会在配置缺失时静默使用默认值。

## 生成和评估

生成默认算法方案：

```bash
PYTHONPATH=src python3 tools/evaluate_multicore.py \
  --cases-dir artifacts/data -o /tmp/multicore-results --q q3
```

也可以直接调用官方评估器。方案文件已经存在时：

```bash
python3 artifacts/code/multicore_cut_evaluate_problem_3.py \
  artifacts/data/case_001.json PLAN.json \
  --config artifacts/data/config.txt
```

`tools/evaluate_multicore.py` 默认使用：

```text
q1 -> subgraph.q1_algorithm:build_plan
q2 -> subgraph.q2_algorithm:build_plan
q3 -> subgraph.q3_algorithm:build_plan
```

评估结果通常包含 `*_res.json`、`*_trace.json` 和 `*_log.txt`，分别用于完整
指标、Perfetto 时间线和摘要日志。COPY、片上内存、Pipe 顺序、Makespan 及 Q3
Cache 命中都以官方评估器输出为准。
