# ============================================================
# 人工智能工具信息 | AI Tool Information
#   工具名称 (Tool Name)        : GLM-5.2
#   版本/型号 (Version/Model)   : GLM-5.2
#   开发机构/公司 (Developer)    : 智谱AI (Zhipu AI / zai-org)
#   版本颁布日期 (Release Date) : 2026-06-13
#   声明：本程序及代码是在人工智能工具辅助下完成的
# ============================================================
"""运行 Q1 五组消融变体并汇总官方评测器指标。

示例：
    PYTHONPATH=src uv run python tools/run_q1_ablation.py \
        --cases-dir artifacts/data -o results/q1_ablation

输出的 CSV 中每行对应一个 (case, variant, num_cores) 组合，包含：makespan、
speedup、added_bytes、partition_count、boundary_count（局部边界数）、
load_cv 与关键路径诊断。另外写一个 summary.json 给出每种变体在 4 核下的
平均指标 + 相对 full 的 makespan 偏差百分比（按图族分组）。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

# evaluate_multicore 与本脚本同在 tools/ 下，需要把 tools 加到 sys.path 才能
# 直接 ``python tools/run_q1_ablation.py`` 运行而无需安装。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_multicore import CORE_COUNTS, EVALUATOR_TIMEOUT_SECONDS, evaluate_cases, discover_cases


# 五个 Q1 变体 → 对应的 module:callable 规格。
VARIANTS = {
    "full": "subgraph.q1_ablation:build_full",
    "no_semantic": "subgraph.q1_ablation:build_no_semantic",
    "no_adaptive": "subgraph.q1_ablation:build_no_adaptive",
    "no_critical_path": "subgraph.q1_ablation:build_no_critical_path",
    "no_rebalance": "subgraph.q1_ablation:build_no_rebalance",
}


def _boundary_count(graph: dict, node_to_subgraph: dict[str, int], features: dict) -> int:
    """统计方案中“跨子图依赖边”的数量，即子图边界数。"""
    return sum(
        node_to_subgraph.get(str(source)) != node_to_subgraph.get(str(target))
        for source, successors in features["succs"].items()
        for target in successors
    )


def _feature_payload(graph: dict) -> dict:
    """构造只含 succs 的轻量 features，用于边界统计。

    这里复用算法/评测器相同的 COPY 收缩后的 succs 图，避免把 COPY 节点
    也算成边界（COPY 由评测器自动插入，不属于用户方案）。
    """
    from subgraph.algorithm_common import analyze_graph
    features = analyze_graph(graph)
    return {"succs": features.succs}


def _collect_rows(output_root: Path, cases: list[Path], cores: tuple[int, ...]) -> list[dict]:
    """遍历所有 (variant, case, cores) 组合，生成 CSV 行。"""
    rows: list[dict] = []
    for variant in VARIANTS:
        variant_root = output_root / variant
        for case_path in cases:
            case_dir = variant_root / case_path.stem
            aggregate = json.loads((case_dir / "aggregate.json").read_text())
            graph = json.loads(case_path.read_text())
            feature_payload = _feature_payload(graph)
            diagnostics = json.loads((case_dir / "diagnostics.json").read_text())["runs"]["q1"]
            for run in aggregate["runs"]:
                core = run["num_cores"]
                if core not in cores:
                    continue
                metrics = run["problems"]["q1"]["metrics"]
                movement = metrics["data_movement_bytes"]
                # data_movement_bytes 在不同问题下可能是数值或字典；归一成 int。
                if isinstance(movement, dict):
                    movement = movement.get("added_copy_bytes", movement.get("scheduled_copy_bytes", 0))
                diag = diagnostics[str(core)]["algorithm"]
                plan = json.loads(
                    (case_dir / "input" / f"plan_q1_{core}cores.json").read_text()
                )
                rows.append({
                    "case": case_path.stem,
                    "variant": variant,
                    "num_cores": core,
                    "graph_family": diag["graph_pattern"]["family"],
                    "graph_pattern": diag["graph_pattern"]["pattern"],
                    "makespan": metrics["makespan"],
                    "speedup": metrics["speedup"],
                    "added_bytes": movement,
                    "partition_count": diag["partition_count"],
                    "boundary_count": _boundary_count(graph, plan["node_to_subgraph"], feature_payload),
                    "load_cv": diag["load_cv"],
                    "max_load_ratio": diag["max_load_ratio"],
                    "critical_path_length": diag["critical_path_length"],
                    "critical_path_cycles": diag["critical_path_cycles"],
                    "cp_cross_core_edges": diag["critical_path_cross_core_edges"],
                })
    return rows


def _write_summary(rows: list[dict], output: Path) -> None:
    """汇总 4 核下每种变体的平均指标 + 相对 full 的 makespan 偏差。"""
    full = {(r["case"], r["num_cores"]): r["makespan"] for r in rows if r["variant"] == "full"}
    summary: list[dict] = []
    for variant in VARIANTS:
        selected = [r for r in rows if r["variant"] == variant and r["num_cores"] == 4]
        summary.append({
            "variant": variant,
            "avg_speedup": mean(r["speedup"] for r in selected) if selected else None,
            "avg_makespan": mean(r["makespan"] for r in selected) if selected else None,
            "avg_added_bytes": mean(r["added_bytes"] for r in selected) if selected else None,
            "avg_partition_count": mean(r["partition_count"] for r in selected) if selected else None,
            "avg_load_cv": mean(r["load_cv"] for r in selected) if selected else None,
            "mean_delta_makespan_pct": mean(
                100.0 * (r["makespan"] - full[(r["case"], r["num_cores"])])
                / full[(r["case"], r["num_cores"])]
                for r in selected
                if full[(r["case"], r["num_cores"])]
            ) if selected else None,
        })
    # 额外按图族细分：每个图族下每种变体的 4 核相对 full 的偏差百分比。
    families = sorted({r["graph_family"] for r in rows})
    family_delta = []
    for family in families:
        for variant in VARIANTS:
            selected = [r for r in rows if r["variant"] == variant and r["num_cores"] == 4 and r["graph_family"] == family]
            family_delta.append({
                "family": family,
                "variant": variant,
                "mean_delta_makespan_pct": mean(
                    100.0 * (r["makespan"] - full[(r["case"], 4)]) / full[(r["case"], 4)]
                    for r in selected if full[(r["case"], 4)]
                ) if selected else None,
            })
    (output / "summary.json").write_text(json.dumps({"core": 4, "variants": summary, "family_delta": family_delta}, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：跑完五组变体 → 生成 CSV + summary.json。"""
    parser = argparse.ArgumentParser(description="运行 Q1 五组消融实验")
    parser.add_argument("--cases-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("results/q1_ablation"))
    parser.add_argument("--cores", nargs="+", type=int, default=list(CORE_COUNTS))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    # 始终把 1 核加入：它是 speedup 的基线。
    cores = tuple(sorted(set(args.cores) | {1}))
    cases = discover_cases(args.cases_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for variant, spec in VARIANTS.items():
        print(f"[{variant}] evaluating {len(cases)} cases")
        batch = evaluate_cases(
            cases, {"q1": spec}, args.output_dir / variant,
            cores=cores, questions=("q1",), workers=args.workers,
            evaluator_timeout=EVALUATOR_TIMEOUT_SECONDS,
        )
        if batch["failed"]:
            raise RuntimeError(f"{variant}: {batch['failed']} case(s) failed")
    rows = _collect_rows(args.output_dir, cases, cores)
    fields = list(rows[0]) if rows else []
    with (args.output_dir / "ablation_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _write_summary(rows, args.output_dir)
    print(f"wrote {len(rows)} rows to {args.output_dir / 'ablation_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
