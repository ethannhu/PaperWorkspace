"""Run the default Q1--Q3 algorithms through the official evaluators.

The runner discovers and evaluates ``case_*.json`` files under the input
directory.  Each case uses the ``config.txt`` in that same directory, keeps
every individual plan/evaluator result, writes one aggregate JSON file, and
saves a ``speedup.png`` plot.

Example::

    PYTHONPATH=src uv run tools/evaluate_multicore.py \
        --cases-dir artifacts/data \
        -o results/multicore_cases
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


QUESTIONS = ("q1", "q2", "q3")
CORE_COUNTS = (1, 2, 3, 4, 5)
WORKERS = 4
EVALUATOR_TIMEOUT_SECONDS = 600
DEFAULT_ALGORITHMS = {
    "q1": "subgraph.q1_algorithm:build_plan",
    "q2": "subgraph.q2_algorithm:build_plan",
    "q3": "subgraph.q3_algorithm:build_plan",
}


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
    reusable_context: dict[str, Any] | None = None,
) -> AlgorithmResult:
    """Call a plan builder with the common graph/core/scenario interface."""
    parameters = inspect.signature(algorithm).parameters
    kwargs: dict[str, Any] = {}
    if "num_cores" in parameters:
        kwargs["num_cores"] = num_cores
    if "scenario" in parameters:
        kwargs["scenario"] = scenario
    for name, value in (reusable_context or {}).items():
        if name in parameters:
            kwargs[name] = value
    result = algorithm(graph, **kwargs)
    if not isinstance(result, AlgorithmResult):
        raise TypeError("algorithm must return AlgorithmResult")
    return result


def _prepare_reusable_context(
    algorithm: Callable[..., AlgorithmResult],
    graph: dict[str, Any],
) -> dict[str, Any]:
    """Build reusable graph features for algorithms that opt in."""
    parameters = inspect.signature(algorithm).parameters
    if "features" not in parameters:
        return {}
    from subgraph.algorithm_common import analyze_graph

    return {"features": analyze_graph(graph)}


def _run_evaluator(
    evaluator: Path,
    graph: Path,
    plan: Path,
    config: Path,
    output: Path,
    timeout_seconds: int | None = EVALUATOR_TIMEOUT_SECONDS,
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
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"evaluator timed out after {timeout_seconds}s for {evaluator.name}"
        ) from exc
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
    algorithm_specs: dict[str, str],
    config_path: Path,
    output_dir: Path,
    cores: tuple[int, ...] = CORE_COUNTS,
    questions: tuple[str, ...] = QUESTIONS,
    evaluator_timeout: int | None = EVALUATOR_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    graph_path = graph_path.resolve()
    config_path = config_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    simulator_output_dir = output_dir / "output"
    simulator_output_dir.mkdir(parents=True, exist_ok=True)
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    selected_algorithms = {
        question: _load_callable(algorithm_specs[question])
        for question in questions
    }
    reusable_contexts = {
        question: _prepare_reusable_context(algorithm, graph)
        for question, algorithm in selected_algorithms.items()
    }
    repo_root = Path(__file__).resolve().parents[1]
    evaluator_dir = repo_root / "artifacts" / "code"

    runs: list[dict[str, Any]] = []
    diagnostics_runs: dict[str, dict[str, Any]] = {}
    for num_cores in cores:
        plans: dict[str, str] = {}
        for question in questions:
            result = _call_algorithm(
                selected_algorithms[question],
                graph,
                num_cores,
                question,
                reusable_contexts[question],
            )
            plan_path = input_dir / f"plan_{question}_{num_cores}cores.json"
            plan_path.write_text(json.dumps(result.plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            plans[question] = str(plan_path)
            diagnostics_runs.setdefault(question, {})[str(num_cores)] = result.diagnostics

        run: dict[str, Any] = {"num_cores": num_cores, "plans": plans, "problems": {}}
        for question in questions:
            problem = int(question[1:])
            evaluator = evaluator_dir / f"multicore_cut_evaluate_problem_{problem}.py"
            # Keep official evaluator artifacts in output/ and algorithm plans
            # in input/ so the case directory has a stable input/output split.
            result_path = simulator_output_dir / f"result_{question}_{num_cores}cores.json"
            result, stdout = _run_evaluator(
                evaluator,
                graph_path,
                Path(plans[question]),
                config_path,
                result_path,
                evaluator_timeout,
            )
            run["problems"][question] = {
                "result_path": str(result_path),
                "metrics": _metric_snapshot(result),
                "stdout": stdout,
            }
        runs.append(run)

    baseline = {
        question: next(
            item["problems"][question]["metrics"]["makespan"]
            for item in runs
            if item["num_cores"] == 1
        )
        for question in questions
    }
    for run in runs:
        for question in questions:
            key = question
            makespan = run["problems"][key]["metrics"]["makespan"]
            run["problems"][key]["metrics"]["speedup"] = (
                baseline[key] / makespan if makespan else None
            )

    aggregate = {
        "graph": str(graph_path),
        "algorithm": algorithm_specs,
        "questions": list(questions),
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
                "algorithm": algorithm_specs,
                "runs": diagnostics_runs,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return aggregate


def discover_cases(cases_dir: Path) -> list[Path]:
    """Return only input graphs named exactly ``case_<number>.json``.

    Evaluators may leave files such as ``case_001_problem_1_trace.json`` in
    the same directory.  A broad ``case_*.json`` glob would incorrectly treat
    those artifacts as additional graph inputs.
    """
    cases_dir = cases_dir.resolve()
    graph_paths = sorted(
        path
        for path in cases_dir.glob("case_*.json")
        if re.fullmatch(r"case_[0-9]+\.json", path.name)
    )
    if not graph_paths:
        raise FileNotFoundError(
            f"expected at least one case_<number>.json file in {cases_dir}"
        )
    return graph_paths


def evaluate_cases(
    graph_paths: list[Path],
    algorithm_specs: dict[str, str],
    output_dir: Path,
    cores: tuple[int, ...] = CORE_COUNTS,
    questions: tuple[str, ...] = QUESTIONS,
    workers: int = WORKERS,
    evaluator_timeout: int | None = EVALUATOR_TIMEOUT_SECONDS,
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
        case_config = graph_path.parent / "config.txt"
        future = executor.submit(
            evaluate,
            graph_path,
            algorithm_specs,
            case_config,
            case_output_dir,
            cores,
            questions,
            evaluator_timeout,
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
                figure_outputs = [str(plot_speedup(aggregate, case_output_dir))]
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
        "algorithm": algorithm_specs,
        "successful": successful,
        "failed": len(cases) - successful,
        "workers": workers,
        "core_counts": list(cores),
        "questions": list(questions),
        "cases": cases,
    }


def plot_speedup(aggregate: dict[str, Any], output_dir: Path) -> Path:
    """Draw measured speedup trends over the single-core baseline."""
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
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
        problem_runs = [item for item in runs if problem in item["problems"]]
        problem_cores = [item["num_cores"] for item in problem_runs]
        values = [item["problems"][problem]["metrics"]["speedup"] for item in problem_runs]
        if values:
            ax.plot(
                problem_cores,
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

    png_path = output_dir / "speedup.png"
    try:
        fig.savefig(png_path, dpi=600, bbox_inches="tight")
    finally:
        plt.close(fig)
    return png_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="批量运行 Q1-Q3 多核算法评测",
    )
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "data",
        help="case 输入目录；默认 artifacts/data",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="输出目录；批量模式默认 results/multicore_cases",
    )
    parser.add_argument(
        "--cores",
        nargs="+",
        type=int,
        default=list(CORE_COUNTS),
        help="核数；默认包含 1-5 核，其中 1 核作为加速比基线",
    )
    parser.add_argument(
        "--q",
        nargs="+",
        choices=QUESTIONS,
        default=list(QUESTIONS),
        help="要评测的问题；可单选或多选，例如 --q q1 q3",
    )
    parser.add_argument(
        "--experimental-search",
        action="store_true",
        help="Q1 使用官方 oracle 引导的种群搜索入口（实验性）",
    )
    args = parser.parse_args(argv)
    if 1 not in args.cores:
        args.cores = [1, *args.cores]

    cores = tuple(sorted(set(args.cores)))
    questions = tuple(args.q)
    algorithm_specs = dict(DEFAULT_ALGORITHMS)
    if args.experimental_search:
        if "q1" not in questions:
            parser.error("--experimental-search requires --q q1")
        algorithm_specs["q1"] = "subgraph.experimental_search:build_plan"
    output_dir = args.output_dir or Path("results/multicore_cases")
    graph_paths = discover_cases(args.cases_dir)
    batch = evaluate_cases(
        graph_paths,
        algorithm_specs,
        output_dir,
        cores,
        questions,
        WORKERS,
        EVALUATOR_TIMEOUT_SECONDS,
    )
    return 1 if batch["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
