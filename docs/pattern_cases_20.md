# 20 个代表性测试用例

选择依据：覆盖 `case_graph_patterns.md` 的全部 7 类算子排布模式；大类抽取多个样本，小类全部或大部分保留，避免只取 `case_001` 到 `case_020` 造成模式偏置。

按编号升序的数字序号：

1. `case_001`
2. `case_003`
3. `case_004`
4. `case_006`
5. `case_007`
6. `case_010`
7. `case_011`
8. `case_016`
9. `case_020`
10. `case_024`
11. `case_036`
12. `case_044`
13. `case_047`
14. `case_052`
15. `case_058`
16. `case_062`
17. `case_068`
18. `case_086`
19. `case_090`
20. `case_100`

| 模式 | 测试用例 |
|---|---|
| 小/中型混合 MLP-Reduce | case_006, case_010, case_052, case_100 |
| 浅层宽并行 MatMul-ReLU | case_001, case_036, case_062 |
| CNN / Residual 卷积块 | case_020, case_044, case_090 |
| Gated / Sigmoid-MLP | case_004, case_007 |
| Attention / Normalize 复杂链 | case_003, case_047, case_068, case_086 |
| 极窄超深 Reduce-Relu-Add | case_016, case_024 |
| 极宽 MatMul-Add 批处理 | case_011, case_058 |

本清单只记录代表性样例；实际 JSON 位于 `artifacts/data/` 和 `artifacts/excases/`，仓库当前没有单独的 `pattern_cases_20.txt` 文件。
