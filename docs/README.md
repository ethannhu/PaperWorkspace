# 文档导航

本目录分为两部分：

- 根目录文档：描述当前仓库中已经实现并经过测试的代码；
- [`ideas/`](ideas/)：保留可能可行的模型、实验方向和未实现设想，不代表当前算法行为。

## 当前实现

- [`algorithm_framework.md`](algorithm_framework.md)：三个问题的入口和总体流程。
- [`challenge_data_contract.md`](challenge_data_contract.md)：输入图、方案 JSON、配置和评估命令。
- [`partition.md`](partition.md)：当前语义分块器的实际规则。
- [`case_graph_patterns.md`](case_graph_patterns.md)：七类图模式及代码中的分类规则。
- [`mixed_algorithm.md`](mixed_algorithm.md)：MIXED 图分支的实现说明。
- [`pattern_cases_20.md`](pattern_cases_20.md)：代表性样例清单。
- [`semantic_analyzer.md`](semantic_analyzer.md)：语义分析工具。
- [`graph_visualizer.md`](graph_visualizer.md)：图可视化工具。

## 阅读边界

`challenge.docx` 和 `artifacts/` 是题面、官方评估器及原始数据；本文档不替代
它们。若文档与代码或评估器冲突，以代码和评估器为准。
