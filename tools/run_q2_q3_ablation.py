# ============================================================
# 人工智能工具信息 | AI Tool Information
#   工具名称 (Tool Name)        : GLM-5.2
#   版本/型号 (Version/Model)   : GLM-5.2
#   开发机构/公司 (Developer)    : 智谱AI (Zhipu AI / zai-org)
#   版本颁布日期 (Release Date) : 2026-06-13
#   声明：本程序及代码是在人工智能工具辅助下完成的
# ============================================================
"""通过官方多核评测器批量运行 Q2 或 Q3 五组消融变体。

示例：
    PYTHONPATH=src uv run python tools/run_q2_q3_ablation.py --q q3

输出 CSV 的列含 Q2/Q3 特有的缓存与跨核流量指标：
    * added_ddr_bytes     —— 评测器计的 DDR 字节；
    * cross_core_edges    —— 评测器输出的 task_dependencies 数量；
    * cross_core_bytes    —— 评测器输出的 cross_task_traffic；
    * cache_hits/misses/hit_rate、hit/miss_bytes、l2_reuse；
    * effective_core_count（短副本链可能小于 num_cores）。
同时给出 4 核下各变体相对 full 的 makespan 偏差百分比。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

# 本脚本与 evaluate_multicore.py 同在 tools/ 下，需要把 tools 加到 sys.path。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_multicore import CORE_COUNTS, EVALUATOR_TIMEOUT_SECONDS, discover_cases, evaluate_cases

# Q2/Q3 五组变体 → module:callable 规格。
SPECS = {
    "q2": {
        "full": "subgraph.q2_ablation:build_full",
        "no_structure": "subgraph.q2_ablation:build_no_structure",
        "no_communication": "subgraph.q2_ablation:build_no_communication",
        "no_critical_path": "subgraph.q2_ablation:build_no_critical_path",
        "no_core_control": "subgraph.q2_ablation:build_no_core_control",
    },
    "q3": {
        "full": "subgraph.q3_ablation:build_full",
        "no_l2": "subgraph.q3_ablation:build_no_l2",
        "no_cache_priority": "subgraph.q3_ablation:build_no_cache_priority",
        "global_cache_priority": "subgraph.q3_ablation:build_global_cache_priority",
        "no_communication": "subgraph.q3_ablation:build_no_communication",
    },
}


def _movement_bytes(result: dict) -> int:
    """把评测器的 data_movement_bytes 字段统一为 int。

    旧版本是数值，新版本是字典（含 added_copy_bytes / scheduled_copy_bytes）。
    """
    movement = result.get("data_movement_bytes", 0)
    if isinstance(movement, dict):
        return int(movement.get("added_copy_bytes", movement.get("scheduled_copy_bytes", 0)))
    return int(movement or 0)


def collect(question: str, root: Path, cases: list[Path], cores: tuple[int, ...]) -> list[dict]:
    """遍历每个变体的所有 case × core，收集 CSV 行。"""
    rows = []
    for variant in SPECS[question]:
        variant_root = root / variant
        for case in cases:
            case_dir = variant_root / case.stem
            aggregate = json.loads((case_dir / "aggregate.json").read_text())
            diagnostics = json.loads((case_dir / "diagnostics.json").read_text())["runs"][question]
            for run in aggregate["runs"]:
                core = run["num_cores"]
                if core not in cores:
                    continue
                metrics = run["problems"][question]["metrics"]
                # 评测器 result JSON 中含 cache_stats / task_dependencies /
                # cross_task_traffic，需要单独读取以填缓存与跨核统计列。
                result = json.loads(Path(run["problems"][question]["result_path"]).read_text())
                run_diag = diagnostics[str(core)]
                diag = run_diag["algorithm"]
                graph_info = run_diag["graph_pattern"]
                cache = result.get("cache_stats", {})
                rows.append({
                    "case": case.stem,
                    "variant": variant,
                    "num_cores": core,
                    "graph_family": graph_info["family"],
                    "graph_pattern": graph_info["pattern"],
                    "makespan": metrics["makespan"],
                    "speedup": metrics["speedup"],
                    "added_ddr_bytes": _movement_bytes(result),
                    "partition_count": diag["partition_count"],
                    "cross_core_edges": len(result.get("task_dependencies", [])),
                    "cross_core_bytes": result.get("cross_task_traffic", 0),
                    "load_cv": diag.get("load_cv", 0.0),
                    "max_load_ratio": diag.get("max_load_ratio", 0.0),
                    "critical_path_length": diag.get("critical_path_length", 0),
                    "critical_path_cycles": diag.get("critical_path_cycles", 0),
                    "cp_cross_core_edges": diag.get("critical_path_cross_core_edges", 0),
                    "effective_core_count": diag.get("effective_core_count", core),
                    "cache_hits": cache.get("hits", 0),
                    "cache_misses": cache.get("accesses", 0) - cache.get("hits", 0),
                    "cache_hit_rate": cache.get("hit_rate", 0.0),
                    "cache_hit_bytes": cache.get("hit_bytes", 0),
                    "cache_miss_bytes": cache.get("miss_bytes", 0),
                    "l2_reuse": diag.get("l2_reuse", True),
                })
    return rows


def write_summary(question: str, rows: list[dict], output: Path) -> None:
    """汇总 4 核下每种变体相对 full 的平均指标。"""
    full = {(r["case"], r["num_cores"]): r["makespan"] for r in rows if r["variant"] == "full"}
    summaries = []
    for variant in SPECS[question]:
        selected = [r for r in rows if r["variant"] == variant and r["num_cores"] == 4]
        summaries.append({
            "variant": variant,
            "avg_speedup": mean(r["speedup"] for r in selected) if selected else None,
            "avg_makespan": mean(r["makespan"] for r in selected) if selected else None,
            "avg_added_ddr_bytes": mean(r["added_ddr_bytes"] for r in selected) if selected else None,
            "avg_cross_core_bytes": mean(r["cross_core_bytes"] for r in selected) if selected else None,
            "avg_load_cv": mean(r["load_cv"] for r in selected) if selected else None,
            "avg_cache_hit_rate": mean(r["cache_hit_rate"] for r in selected) if selected else None,
            "mean_delta_makespan_pct": mean(
                100 * (r["makespan"] - full[(r["case"], 4)]) / full[(r["case"], 4)]
                for r in selected if full[(r["case"], 4)]
            ) if selected else None,
        })
    (output / "summary.json").write_text(json.dumps({"question": question, "core": 4, "variants": summaries}, indent=2) + "\n")


def main(argv=None) -> int:
    """CLI 入口：选 q2/q3 → 跑完五组变体 → CSV + summary.json。"""
    parser = argparse.ArgumentParser(description="运行 Q2/Q3 消融实验")
    parser.add_argument("--q", choices=("q2", "q3"), required=True)
    parser.add_argument("--cases-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument("-o", "--output-dir", type=Path)
    parser.add_argument("--cores", nargs="+", type=int, default=list(CORE_COUNTS))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    output = args.output_dir or Path(f"results/{args.q}_ablation")
    output.mkdir(parents=True, exist_ok=True)
    cores = tuple(sorted(set(args.cores) | {1}))
    cases = discover_cases(args.cases_dir)
    for variant, spec in SPECS[args.q].items():
        print(f"[{args.q}/{variant}] evaluating {len(cases)} cases")
        batch = evaluate_cases(
            cases, {args.q: spec}, output / variant, cores=cores,
            questions=(args.q,), workers=args.workers,
            evaluator_timeout=EVALUATOR_TIMEOUT_SECONDS,
        )
        if batch["failed"]:
            raise RuntimeError(f"{variant}: {batch['failed']} case(s) failed")
    rows = collect(args.q, output, cases, cores)
    csv_path = output / f"{args.q}_ablation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_summary(args.q, rows, output)
    print(f"wrote {len(rows)} rows to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
