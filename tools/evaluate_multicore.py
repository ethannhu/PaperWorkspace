"""Run one partition/scheduling algorithm through the official Q1--Q3 evaluators.

The algorithm is supplied as ``module:callable`` and must return the standard
multicore plan dictionary.  The runner evaluates one-core baseline plus 2--5
cores, keeps every individual plan/evaluator result, and writes one aggregate
JSON file.  When no graph is supplied, the runner discovers and evaluates the
100 ``case_*.json`` files under ``artifacts/data`` and displays case-level
progress with tqdm; batch mode writes only each case's own artifacts and no
cross-case summary.  Plotting is intentionally a separate backend-specific
step.

Example::

    PYTHONPATH=src uv run tools/evaluate_multicore.py \
        artifacts/data/case_001.json \
        --algorithm subgraph.demo_framework:build_plan \
        --config artifacts/data/config.txt \
        -o results/case_001

    PYTHONPATH=src uv run tools/evaluate_multicore.py \
        --algorithm subgraph.demo_framework:build_plan
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from subgraph.interfaces import AlgorithmResult


PROBLEMS = (1, 2, 3)
CORE_COUNTS = (1, 2, 3, 4, 5)
DEFAULT_CASE_COUNT = 100
DEFAULT_WORKERS = 4


def _load_callable(spec: str) -> Callable[..., AlgorithmResult]:
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
    algorithm: Callable[..., AlgorithmResult],
    graph: dict[str, Any],
    num_cores: int,
    scenario: str,
) -> AlgorithmResult:
    """Call a plan builder with the common graph/core/scenario interface."""
    parameters = inspect.signature(algorithm).parameters
    kwargs: dict[str, Any] = {}
    if "num_cores" in parameters:
        kwargs["num_cores"] = num_cores
    if "scenario" in parameters:
        kwargs["scenario"] = scenario
    result = algorithm(graph, **kwargs)
    if not isinstance(result, AlgorithmResult):
        raise TypeError("algorithm must return AlgorithmResult")
    return result


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
    input_dir = output_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    simulator_output_dir = output_dir / "output"
    simulator_output_dir.mkdir(parents=True, exist_ok=True)
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    algorithm = _load_callable(algorithm_spec)
    repo_root = Path(__file__).resolve().parents[1]
    evaluator_dir = repo_root / "artifacts" / "code"

    runs: list[dict[str, Any]] = []
    diagnostics_runs: dict[str, dict[str, Any]] = {}
    for num_cores in cores:
        plans: dict[str, str] = {}
        for problem in problems:
            scenario = f"q{problem}"
            result = _call_algorithm(
                algorithm,
                graph,
                num_cores,
                scenario,
            )
            plan_path = input_dir / f"plan_{scenario}_{num_cores}cores.json"
            plan_path.write_text(json.dumps(result.plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            plans[scenario] = str(plan_path)
            diagnostics_runs.setdefault(scenario, {})[str(num_cores)] = result.diagnostics

        run: dict[str, Any] = {"num_cores": num_cores, "plans": plans, "problems": {}}
        for problem in problems:
            scenario = f"q{problem}"
            evaluator = evaluator_dir / f"multicore_cut_evaluate_problem_{problem}.py"
            # Keep official evaluator artifacts in output/ and algorithm plans
            # in input/ so the case directory has a stable input/output split.
            result_path = simulator_output_dir / f"result_{scenario}_{num_cores}cores.json"
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
    diagnostics_path = output_dir / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(
            {
                "algorithm": algorithm_spec,
                "runs": diagnostics_runs,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return aggregate


def discover_cases(cases_dir: Path, case_count: int = DEFAULT_CASE_COUNT) -> list[Path]:
    """Return only input graphs named exactly ``case_<number>.json``.

    Evaluators may leave files such as ``case_001_problem_1_trace.json`` in
    the same directory.  A broad ``case_*.json`` glob would incorrectly treat
    those artifacts as additional graph inputs.
    """
    if case_count < 1:
        raise ValueError("case_count must be at least 1")
    cases_dir = cases_dir.resolve()
    graph_paths = sorted(
        path
        for path in cases_dir.glob("case_*.json")
        if re.fullmatch(r"case_[0-9]+\.json", path.name)
    )
    if len(graph_paths) < case_count:
        raise FileNotFoundError(
            f"expected at least {case_count} case_<number>.json files in {cases_dir}, "
            f"found {len(graph_paths)}"
        )
    return graph_paths[:case_count]


def evaluate_cases(
    graph_paths: list[Path],
    algorithm_spec: str,
    output_dir: Path,
    config_path: Path | None = None,
    cores: tuple[int, ...] = CORE_COUNTS,
    problems: tuple[int, ...] = PROBLEMS,
    make_plots: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    """Evaluate cases concurrently, keeping only per-case output artifacts."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    case_results: list[dict[str, Any] | None] = [None] * len(graph_paths)

    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {}
    for index, graph_path in enumerate(graph_paths):
        case_output_dir = output_dir / graph_path.stem
        case_config = config_path or graph_path.parent / "config.txt"
        future = executor.submit(
            evaluate,
            graph_path,
            algorithm_spec,
            case_config,
            case_output_dir,
            cores,
            problems,
        )
        futures[future] = (index, graph_path, case_output_dir)

    progress = tqdm(
        total=len(futures),
        desc="Evaluating cases",
        unit="case",
        dynamic_ncols=True,
    )
    try:
        for future in as_completed(futures):
            index, graph_path, case_output_dir = futures[future]
            progress.set_postfix_str(graph_path.stem)
            try:
                aggregate = future.result()
                figure_outputs = (
                    [str(path) for path in plot_speedup(aggregate, case_output_dir)]
                    if make_plots
                    else []
                )
                case_results[index] = {
                    "case": graph_path.stem,
                    "graph": str(graph_path.resolve()),
                    "status": "ok",
                    "aggregate": str(case_output_dir / "aggregate.json"),
                    "baseline_makespan": aggregate["baseline_makespan"],
                    "figure_outputs": figure_outputs,
                }
            except Exception as exc:  # Keep the remaining cases evaluable.
                case_results[index] = {
                    "case": graph_path.stem,
                    "graph": str(graph_path.resolve()),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                progress.update()
    except KeyboardInterrupt:
        # Do not wait for the remaining subprocesses after Ctrl-C.  Completed
        # cases and their per-case artifacts are already persisted above.
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    finally:
        progress.close()

    cases = [item for item in case_results if item is not None]
    successful = sum(item["status"] == "ok" for item in cases)
    return {
        "algorithm": algorithm_spec,
        "case_count": len(cases),
        "requested_case_count": len(graph_paths),
        "successful": successful,
        "failed": len(cases) - successful,
        "workers": workers,
        "core_counts": list(cores),
        "problems": list(problems),
        "cases": cases,
    }


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
    # fig.savefig(svg_path, bbox_inches="tight")
    # fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    # fig.savefig(tiff_path, dpi=600, bbox_inches="tight")
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
    parser = argparse.ArgumentParser(
        description="运行 Q1-Q3 多核算法评测；省略 graph 时自动评测 100 个 cases"
    )
    parser.add_argument(
        "graph",
        type=Path,
        nargs="?",
        help="单个计算图 JSON；省略时进入 100 cases 批量模式",
    )
    parser.add_argument(
        "--algorithm",
        default="subgraph.demo_framework:build_plan",
        help="算法入口 module:callable，默认使用 demo_framework:build_plan",
    )
    parser.add_argument("--config", type=Path, help="评测 config.txt")
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "data",
        help="批量模式的 case 目录；默认 artifacts/data",
    )
    parser.add_argument(
        "--case-count",
        type=int,
        default=DEFAULT_CASE_COUNT,
        help="批量模式评测的 case 数；默认 100",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"批量模式并发线程数；默认 {DEFAULT_WORKERS}",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="输出目录；批量模式默认 results/multicore_100cases",
    )
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
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        default=True,
        help="生成 speedup 图；单 case 模式默认开启，批量模式默认关闭",
    )
    plot_group.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="只运行评测，不生成 speedup.svg/pdf/png",
    )
    args = parser.parse_args(argv)
    if 1 not in args.cores:
        args.cores = [1, *args.cores]

    cores = tuple(sorted(set(args.cores)))
    problems = tuple(args.problems)
    if args.graph is None:
        output_dir = args.output_dir or Path("results/multicore_100cases")
        graph_paths = discover_cases(args.cases_dir, args.case_count)
        batch = evaluate_cases(
            graph_paths,
            args.algorithm,
            output_dir,
            args.config,
            cores,
            problems,
            make_plots=args.plot is True,
            workers=args.workers,
        )
        return 1 if batch["failed"] else 0

    output_dir = args.output_dir or Path("results") / args.graph.stem
    config = args.config or args.graph.parent / "config.txt"
    aggregate = evaluate(
        args.graph,
        args.algorithm,
        config,
        output_dir,
        cores,
        problems,
    )
    make_plot = args.plot is not False
    figure_outputs = [] if not make_plot else [str(path) for path in plot_speedup(aggregate, output_dir)]
    print(json.dumps({
        "aggregate": str(output_dir / "aggregate.json"),
        "runs": len(aggregate["runs"]),
        "baseline_makespan": aggregate["baseline_makespan"],
        "figure_outputs": figure_outputs,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
