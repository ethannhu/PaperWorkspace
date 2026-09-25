# 算子语义特征分析器

`tools/semantic_analyzer.py` 只分析原始计算图，不生成多核方案，也不模拟评估器。

## 使用

```bash
python3 tools/semantic_analyzer.py \
  artifacts/data/case_001.json \
  -o /tmp/case_001_semantic_features.json
```

输出 JSON 包含：

- `operators`：每个算子的角色、语义 family、资源类型、流水线、周期、输入输出张量、数据量、依赖深度、分支/汇聚信息；
- `semantic_edges`：跳过 COPY 节点后的算子依赖边，包含融合分数、融合/切分建议和通信数据量；
- `graph`：全图计数、拓扑序、角色分布和候选融合/切分边数量。

标签规则直接使用数据中的 `op` 和 `pipe`：`MATMUL/CONV` 识别为矩阵计算，`REDUCE` 识别为归约，常见逐元素算子识别为 elementwise，COPY 识别为通信。该工具的分数用于分析和排查，不等同于 `semantic_partition.py` 的最终合并决策。
