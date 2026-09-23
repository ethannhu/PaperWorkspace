# 计算图可视化工具

该工具读取 `artifacts/data/case_*.json`，用于快速理解计算图的规模、算子组成、存储层级、流水线分布和依赖路径。

## 启动

在仓库根目录执行：

```bash
uv run python tools/graph_visualizer.py
```

默认访问 <http://127.0.0.1:8765>。如果当前环境不能自动打开浏览器，可使用：

```bash
uv run python tools/graph_visualizer.py --no-browser
```

## 交互说明

- 选择案例后，顶部指标和两个分布图会立即更新；
- “聚焦算子”默认选择计算周期较高的非 COPY 算子；
- “依赖邻域”按边数展开聚焦点附近的 Tensor/Operator 子图；
- 节点颜色：算子按 `pipe` 着色，张量按 `pos`（DDR/L1/UB）着色；
- 可点击图中的算子切换聚焦点；
- 大图默认限制显示 240 个节点，避免把完整的数万节点图一次性渲染到浏览器。

后端只使用 Python 标准库，前端使用浏览器原生 SVG，因此不需要安装额外依赖。
