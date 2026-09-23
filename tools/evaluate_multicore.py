"""Run one partition/scheduling algorithm through the official Q1--Q3 evaluators.

The algorithm is supplied as ``module:callable`` and must return the standard
multicore plan dictionary.  The runner evaluates one-core baseline plus 2--5
cores, keeps every individual plan/evaluator result, and writes one aggregate
JSON file.  Plotting is intentionally a separate backend-specific step.

Example::

    PYTHONPATH=src python tools/evaluate_multicore.py \
        artifacts/data/case_001.json \
        --algorithm subgraph.demo_framework:build_plan \
        --config artifacts/data/config.txt \
        -o results/case_001
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


PROBLEMS = (1, 2, 3)
CORE_COUNTS = (1, 2, 3, 4, 5)


def _load_callable(spec: str) -> Callable[..., dict[str, Any]]:
    if ":" not in spec:
        raise ValueError("algorithm must use module:callable syntax")
    module_name, function_name = spec.split(":", 1)
    if not module_name or not function_name:
        raise ValueError("algorithm must use module:callable syntax")
    module = importlib.import_module(module_name)
    algorithm = getattr(module, function_name, None)
    if not callable(algorithm):
        raise TypeError(f"algorithm is not callable: {spec}")
    return algorithm


def _call_algorithm(
    algorithm: Callable[..., dict[str, Any]],
    graph: dict[str, Any],
    num_cores: int,
    scenario: str,
) -> dict[str, Any]:
    """Call algorithms with the common interface, allowing simple baselines."""
    parameters = inspect.signature(algorithm).parameters
    kwargs: dict[str, Any] = {}
    if "num_cores" in parameters:
        kwargs["num_cores"] = num_cores
    if "scenario" in parameters:
        kwargs["scenario"] = scenario
    plan = algorithm(graph, **kwargs)
    if not isinstance(plan, dict):
        raise TypeError("algorithm must return a plan object")
    return plan


def _run_evaluator(
    evaluator: Path,
    graph: Path,
    plan: Path,
    config: Path,
    output: Path,
) -> tuple[dict[str, Any], str]:
    trace = output.with_name(output.stem + "_trace.json")
    log = output.with_name(output.stem + "_log.txt")
    command = [
        sys.executable,
        str(evaluator),
        str(graph),
        str(plan),
        "--config",
        str(config),
        "-o",
        str(output),
        "--trace-output",
        str(trace),
        "--log-output",
        str(log),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"evaluator failed for {evaluator.name}: {details}")
    with output.open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    return result, completed.stdout.strip()


def _metric_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    snapshot = {
        "makespan": result.get("makespan"),
        "num_cores": result.get("num_cores"),
        "data_movement_bytes": result.get("data_movement_bytes"),
        "cross_task_traffic": result.get("cross_task_traffic"),
    }
    cache = result.get("cache_stats")
    if isinstance(cache, dict):
        snapshot["cache_stats"] = {
            key: cache.get(key)
            for key in ("hits", "accesses", "hit_bytes", "miss_bytes", "hit_rate")
            if key in cache
        }
    return snapshot


def evaluate(
    graph_path: Path,
    algorithm_spec: str,
    config_path: Path,
    output_dir: Path,
    cores: tuple[int, ...] = CORE_COUNTS,
    problems: tuple[int, ...] = PROBLEMS,
) -> dict[str, Any]:
    graph_path = graph_path.resolve()
    config_path = config_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    algorithm = _load_callable(algorithm_spec)
    repo_root = Path(__file__).resolve().parents[1]
    evaluator_dir = repo_root / "artifacts" / "code"

    runs: list[dict[str, Any]] = []
    for num_cores in cores:
        plans: dict[str, str] = {}
        for problem in problems:
            scenario = f"q{problem}"
            plan = _call_algorithm(algorithm, graph, num_cores, scenario)
            plan_path = output_dir / f"plan_{scenario}_{num_cores}cores.json"
            plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            plans[scenario] = str(plan_path)

        run: dict[str, Any] = {"num_cores": num_cores, "plans": plans, "problems": {}}
        for problem in problems:
            scenario = f"q{problem}"
            evaluator = evaluator_dir / f"multicore_cut_evaluate_problem_{problem}.py"
            result_path = output_dir / f"result_{scenario}_{num_cores}cores.json"
            result, stdout = _run_evaluator(
                evaluator,
                graph_path,
                Path(plans[scenario]),
                config_path,
                result_path,
            )
            run["problems"][scenario] = {
                "result_path": str(result_path),
                "metrics": _metric_snapshot(result),
                "stdout": stdout,
            }
        runs.append(run)

    baseline = {
        f"q{problem}": next(
            item["problems"][f"q{problem}"]["metrics"]["makespan"]
            for item in runs
            if item["num_cores"] == 1
        )
        for problem in problems
    }
    for run in runs:
        for problem in problems:
            key = f"q{problem}"
            makespan = run["problems"][key]["metrics"]["makespan"]
            run["problems"][key]["metrics"]["speedup"] = (
                baseline[key] / makespan if makespan else None
            )

    aggregate = {
        "graph": str(graph_path),
        "algorithm": algorithm_spec,
        "config": str(config_path),
        "core_counts": list(cores),
        "baseline_core_count": 1,
        "baseline_makespan": baseline,
        "runs": runs,
    }
    aggregate_path = output_dir / "aggregate.json"
    aggregate_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return aggregate


def plot_speedup(aggregate: dict[str, Any], output_dir: Path) -> list[Path]:
    """Draw the Q1--Q3 speedup trend and export editable/vector formats."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    runs = sorted(aggregate["runs"], key=lambda item: item["num_cores"])
    core_counts = [item["num_cores"] for item in runs]
    colors = {"q1": "#0F4D92", "q2": "#42949E", "q3": "#B64342"}
    labels = {"q1": "Q1", "q2": "Q2", "q3": "Q3"}

    fig, ax = plt.subplots(figsize=(3.5, 2.7), constrained_layout=False)
    for problem in ("q1", "q2", "q3"):
        values = [
            item["problems"][problem]["metrics"]["speedup"]
            for item in runs
            if problem in item["problems"]
        ]
        if not values:
            continue
        ax.plot(
            core_counts[: len(values)],
            values,
            marker="o",
            markersize=4,
            linewidth=1.7,
            color=colors[problem],
            label=labels[problem],
        )
    ax.axhline(1.0, color="#767676", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xlabel("Number of cores")
    ax.set_ylabel("Speedup over 1 core")
    ax.set_xticks(core_counts)
    ax.set_xlim(min(core_counts) - 0.12, max(core_counts) + 0.12)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=3, handlelength=1.8, columnspacing=1.0)
    fig.tight_layout(pad=0.8)

    skill_scripts = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "nature-figure" / "scripts"
    sys.path.insert(0, str(skill_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    base = output_dir / "speedup"
    require_matplotlib_panel_alignment(
        fig,
        json_out=base.with_suffix(".alignment.json"),
        overlay_svg=base.with_suffix(".alignment.svg"),
        strict=True,
    )
    # Keep explicit vector and raster exports so the source audit can verify
    # the delivery bundle without evaluating dynamic suffix construction.
    svg_path = base.with_suffix(".svg")
    pdf_path = base.with_suffix(".pdf")
    png_path = base.with_suffix(".png")
    tiff_path = base.with_suffix(".tiff")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight")
    outputs = [svg_path, pdf_path, png_path, tiff_path]
    plt.close(fig)

    speedup_data = {
        "core_counts": core_counts,
        "series": {
            problem: [
                item["problems"][problem]["metrics"]["speedup"]
                for item in runs
                if problem in item["problems"]
            ]
            for problem in ("q1", "q2", "q3")
        },
    }
    data_path = output_dir / "speedup_data.json"
    data_path.write_text(
        json.dumps(speedup_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    outputs.append(data_path)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行 Q1-Q3 多核算法评测并聚合结果")
    parser.add_argument("graph", type=Path, help="计算图 JSON")
    parser.add_argument(
        "--algorithm",
        default="subgraph.demo_framework:build_plan",
        help="算法入口 module:callable，默认使用 demo_framework:build_plan",
    )
    parser.add_argument("--config", type=Path, help="评测 config.txt")
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cores",
        nargs="+",
        type=int,
        default=list(CORE_COUNTS),
        help="核数；默认包含 1-5 核，其中 1 核作为加速比基线",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        type=int,
        choices=PROBLEMS,
        default=list(PROBLEMS),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="只运行评测，不生成 speedup.svg/pdf/png",
    )
    args = parser.parse_args(argv)
    config = args.config or args.graph.parent / "config.txt"
    if 1 not in args.cores:
        args.cores = [1, *args.cores]
    aggregate = evaluate(
        args.graph,
        args.algorithm,
        config,
        args.output_dir,
        tuple(sorted(set(args.cores))),
        tuple(args.problems),
    )
    figure_outputs = [] if args.no_plot else [str(path) for path in plot_speedup(aggregate, args.output_dir)]
    print(json.dumps({
        "aggregate": str(args.output_dir / "aggregate.json"),
        "runs": len(aggregate["runs"]),
        "baseline_makespan": aggregate["baseline_makespan"],
        "figure_outputs": figure_outputs,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
