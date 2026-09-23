# 算子语义特征分析器

`tools/semantic_analyzer.py` 是总体“语义识别 → partition → buffer/通信 → 调度”框架中的第一部分。它只分析原始计算图，不生成题目要求的多核方案。

## 使用

```bash
uv run python tools/semantic_analyzer.py \
  artifacts/data/case_001.json \
  -o /tmp/case_001_semantic_features.json
```

输出 JSON 包含：

- `operators`：每个算子的角色、语义 family、资源类型、流水线、周期、输入输出张量、数据量、依赖深度、分支/汇聚信息；
- `semantic_edges`：跳过 COPY 节点后的算子依赖边，包含融合分数、融合/切分建议和通信数据量；
- `graph`：全图计数、拓扑序、角色分布和候选融合/切分边数量。

初版标签规则直接使用数据中的 `op` 和 `pipe`：`MATMUL/CONV` 识别为矩阵计算，`REDUCE` 识别为归约，常见逐元素算子识别为 elementwise，COPY 识别为通信。后续 partition 算法可以基于 `semantic_edges[].decision` 和 `score` 进行聚类。
