# ============================================================
# 人工智能工具信息 | AI Tool Information
#   工具名称 (Tool Name)        : GLM-5.2
#   版本/型号 (Version/Model)   : GLM-5.2
#   开发机构/公司 (Developer)    : 智谱AI (Zhipu AI / zai-org)
#   版本颁布日期 (Release Date) : 2026-06-13
#   声明：本程序及代码是在人工智能工具辅助下完成的
# ============================================================
"""用官方评测器批量运行 Q1--Q3 算法的主驱动。

本runner 会在输入目录下发现并评估 ``case_*.json``。每个 case 使用同一目
录下的 ``config.txt``；保留每份 plan 与评测器结果，写一个 aggregate JSON，
并保存一张 ``speedup.png`` 图。

示例：
    PYTHONPATH=src uv run tools/evaluate_multicore.py \
        --cases-dir artifacts/data \
        -o results/multicore_cases

并发模型：``evaluate_cases`` 用 ``ThreadPoolExecutor`` 把每个 case 作为一个
任务跑（每个 case 内部仍顺序跑 1~5 核 × 3 题）；Ctrl-C 会立刻取消未完成的
任务，已完成的 case 与产物仍然落盘。
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


# 默认全题 + 1~5 核 + 4 进程并发的常量。可以通过 CLI 覆盖。
QUESTIONS = ("q1", "q2", "q3")
CORE_COUNTS = (1, 2, 3, 4, 5)
WORKERS = 4
EVALUATOR_TIMEOUT_SECONDS = 600
# 默认算法规格：直接指向 ``src/subgraph`` 中各题的生产入口。
DEFAULT_ALGORITHMS = {
    "q1": "subgraph.q1_algorithm:build_plan",
    "q2": "subgraph.q2_algorithm:build_plan",
    "q3": "subgraph.q3_algorithm:build_plan",
}


def _load_callable(spec: str) -> Callable[..., AlgorithmResult]:
    """解析 ``module:callable`` 规格并返回对应函数。"""
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
    """用统一的 graph/core/scenario 接口调用 plan builder。

    这里通过 ``inspect`` 探测算法签名：只有当函数声明了 ``num_cores``/
    ``scenario``/``features`` 等参数时才传它。这样不同算法可以省略不关心
    的参数，而不需要强制保持完全一致的签名。
    """
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
    """为声明了 ``features`` 参数的算法预生成一次图特征，便于跨核复用。

    每个 case 会在不同核数下调用同一算法多次；预生成 features 可以省掉重复
    的 ``analyze_graph`` 开销。算法按需取用：没有声明 ``features`` 参数就返回
    空 dict。
    """
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
    """以子进程方式跑一次官方评测器，返回 (result_dict, stdout)。"""
    # trace 与 log 文件名从 output 派生，保持一一对应。
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
    """从评测器 result 中提取需要进入 aggregate 的指标快照。"""
    snapshot = {
        "makespan": result.get("makespan"),
        "num_cores": result.get("num_cores"),
        "data_movement_bytes": result.get("data_movement_bytes"),
        "cross_task_traffic": result.get("cross_task_traffic"),
    }
    cache = result.get("cache_stats")
    # cache_stats 只在 Q3 评测器输出里出现；存在则把常用字段挑出来。
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
    """对一张图跑所有 (核数 × 问题) 组合，写出 aggregate.json / diagnostics.json。

    目录结构：
        * ``input/plan_<question>_<cores>cores.json`` —— 算法方案；
        * ``output/result_<question>_<cores>cores.json`` —— 评测器结果（含
          trace 与 log 副产物）；
        * ``aggregate.json`` —— 整体结果汇总 + speedup；
        * ``diagnostics.json`` —— 算法诊断字段（每核每题一份）。
    speedup 的基线固定是 1 核 makespan。
    """
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
    # 每个 question 都按需预生成 reusable_context（最常见是 features）。
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
            # 把 graph / num_cores / scenario / features（如果有）按签名喂进去。
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
            # 把官方评测器产物放 output/、算法方案放 input/，保持输入/输出分离。
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

    # speedup 基线 = 1 核 makespan；任何 case 都必须有 1 核结果。
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
    """只返回名字严格为 ``case_<number>.json`` 的输入图。

    评测器会在同目录下留下 ``case_001_problem_1_trace.json`` 这类文件。宽
    松的 ``case_*.json`` glob 会把它们错误地当成额外的输入图。
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
    """并发评估多个 case，每个 case 只保留它自己的输出产物。

    Ctrl-C 会立刻取消未完成的 future；已成功的 case 与其产物已经落盘，不受
    影响。失败 case 不会阻断其他 case，只在最终 batch 里被打上 status=failed。
    """
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
            # set_postfix_str 让进度条尾显示当前 case 名，便于排查卡住的 case。
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
            except Exception as exc:  # 保留其余 case 仍可评测。
                case_results[index] = {
                    "case": graph_path.stem,
                    "graph": str(graph_path.resolve()),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                progress.update()
    except KeyboardInterrupt:
        # Ctrl-C 后不等剩余子进程；已完成 case 的产物上面已经落盘。
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
    """绘制并保存 speedup 趋势图（相对单核基线）。

    使用 ``Agg`` 后端避免依赖 X server；样式表刻意设为论文风格：去除顶/右框
    线、字号 8 pt、低饱和度配色。每条线一种问题，Q1/Q2/Q3 用固定颜色。
    """
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
    # 三题固定配色与标签，便于跨图对比。
    colors = {"q1": "#0F4D92", "q2": "#42949E", "q3": "#B64342"}
    labels = {"q1": "Q1", "q2": "Q2", "q3": "Q3"}

    fig, ax = plt.subplots(figsize=(3.5, 2.7), constrained_layout=False)
    for problem in ("q1", "q2", "q3"):
        # 跳过当前 aggregate 里没有的题目，避免空线。
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
    # 1.0 处的虚线：speedup 基准，便于目测是否退化。
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

    # dpi=600 满足论文级要求；bbox_inches=tight 去掉多余白边。
    png_path = output_dir / "speedup.png"
    try:
        fig.savefig(png_path, dpi=600, bbox_inches="tight")
    finally:
        plt.close(fig)
    return png_path


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：解析参数 → 跑 evaluate_cases → 报告成功/失败数。"""
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
    # speedup 基线固定 = 1 核；用户没显式带 1 也强行加上。
    if 1 not in args.cores:
        args.cores = [1, *args.cores]

    cores = tuple(sorted(set(args.cores)))
    questions = tuple(args.q)
    algorithm_specs = dict(DEFAULT_ALGORITHMS)
    if args.experimental_search:
        # 实验性搜索只支持 Q1，避免误用导致 Q2/Q3 跑出意外结果。
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
