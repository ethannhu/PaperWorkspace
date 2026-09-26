# ============================================================
# 人工智能工具信息 | AI Tool Information
#   工具名称 (Tool Name)        : GLM-5.2
#   版本/型号 (Version/Model)   : GLM-5.2
#   开发机构/公司 (Developer)    : 智谱AI (Zhipu AI / zai-org)
#   版本颁布日期 (Release Date) : 2026-06-13
#   声明：本程序及代码是在人工智能工具辅助下完成的
# ============================================================
"""在比赛 case 上批量评估官方随机多核 stub。

对每个 ``case_<number>.json``，先调用
``artifacts/code/stub_multicore_cut_and_schedule.py`` 生成一份 stub 方案，然后
再对每个 (case, core, question) 跑官方评测器。所有产物都落在每 case 一个目
录下，不会修改 ``artifacts/data`` 中的输入。

示例：
    uv run tools/evaluate_stub_multicore.py

默认会跑全部 100 个 case、Q1--Q3、1--5 核。可通过环境变量 ``WORKERS`` 控制
评测器的并发数；stub 方案的生成会先用一个线程池跑完，再启动评测器并发，这
样即便某次评测失败，所有输入方案都已落盘可供排查。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# tqdm 不可用时退化为简单的计数式进度条，保持脚本可独立运行。
try:
    from tqdm import tqdm
except ImportError:  # tqdm 缺失时让脚本仍可用纯 Python 跑。
    class _Progress:
        def __init__(self, total: int, **_: Any) -> None:
            self.total = total
            self.count = 0

        def update(self, amount: int = 1) -> None:
            self.count += amount
            print(f"Progress: {self.count}/{self.total}", file=sys.stderr)

        def close(self) -> None:
            pass

    def tqdm(*, total: int, **kwargs: Any) -> _Progress:
        return _Progress(total, **kwargs)


QUESTIONS = ("q1", "q2", "q3")
CORE_COUNTS = (1, 2, 3, 4, 5)
# 可在运行前用 export WORKERS=8 控制评测器并发。stub 生成阶段先用线程池跑完，
# 评测阶段再开 WORKERS 个并发。
WORKERS = int(os.environ.get("WORKERS", "4"))
DEFAULT_TIMEOUT_SECONDS = 600


def discover_cases(cases_dir: Path) -> list[Path]:
    """返回 case 输入文件，排除评测器中转产出的 trace/result JSON。

    评测器会在同目录下留下 ``case_001_problem_1_trace.json`` 这类文件，朴素
    glob ``case_*.json`` 会把它们误当成输入图。这里用更严格的正则匹配。
    """
    cases_dir = cases_dir.resolve()
    cases = sorted(
        path
        for path in cases_dir.glob("case_*.json")
        if re.fullmatch(r"case_[0-9]+\.json", path.name)
    )
    if not cases:
        raise FileNotFoundError(
            f"expected at least one case_<number>.json file in {cases_dir}"
        )
    return cases


def _run(command: list[str], timeout_seconds: int) -> str:
    """运行一条官方脚本并返回其标准输出的精简版。"""
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"command timed out after {timeout_seconds}s: {command}") from exc
    if completed.returncode:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"command failed: {details}")
    return completed.stdout.strip()


def _metrics(result: dict[str, Any]) -> dict[str, Any]:
    """把评测器 result 中的关键指标挑出来放进 aggregate，不丢细节。"""
    metrics = {
        "makespan": result.get("makespan"),
        "num_cores": result.get("num_cores"),
        "data_movement_bytes": result.get("data_movement_bytes"),
        "cross_task_traffic": result.get("cross_task_traffic"),
    }
    if isinstance(result.get("cache_stats"), dict):
        metrics["cache_stats"] = result["cache_stats"]
    return metrics


def _case_seed(graph_path: Path, base_seed: int) -> int:
    """每个 case 用 ``base_seed + case_<n> 的 n`` 作为种子，保证：

        1. 不同 case 的 stub 不会巧合使用同一随机流；
        2. 同一 case 多次跑结果完全确定，便于回归。
    """
    match = re.fullmatch(r"case_([0-9]+)", graph_path.stem)
    return base_seed + (int(match.group(1)) if match else 0)


def generate_plan(
    graph_path: Path,
    output_dir: Path,
    num_cores: int,
    seed: int,
    min_subgraph_size: int,
    max_subgraph_size: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    """生成并保存一份 stub 方案；这里不调用任何评测器。"""
    graph_path = graph_path.resolve()
    config_path = graph_path.parent / "config.txt"
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / "input"
    input_dir.mkdir(exist_ok=True)
    code_dir = Path(__file__).resolve().parents[1] / "artifacts" / "code"
    stub = code_dir / "stub_multicore_cut_and_schedule.py"
    plan_seed = _case_seed(graph_path, seed)
    plan_path = input_dir / f"plan_{num_cores}cores.json"
    stdout = _run(
        [
            "uv", "run", "python", str(stub), str(graph_path),
            "--num-cores", str(num_cores), "--seed", str(plan_seed),
            "--min-subgraph-size", str(min_subgraph_size),
            "--max-subgraph-size", str(max_subgraph_size),
            "--output", str(plan_path),
        ],
        timeout_seconds,
    )
    return {
        "num_cores": num_cores,
        "plan": str(plan_path),
        "stub_stdout": stdout,
        "problems": {},
    }


def evaluate_plan(
    graph_path: Path,
    plan_path: Path,
    output_dir: Path,
    num_cores: int,
    question: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """对已生成的 stub 方案跑一次官方评测器。"""
    code_dir = Path(__file__).resolve().parents[1] / "artifacts" / "code"
    problem = int(question[1:])
    evaluator = code_dir / f"multicore_cut_evaluate_problem_{problem}.py"
    config_path = graph_path.parent / "config.txt"
    result_dir = output_dir / "output"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"result_{question}_{num_cores}cores.json"
    trace_path = result_dir / f"result_{question}_{num_cores}cores_trace.json"
    log_path = result_dir / f"result_{question}_{num_cores}cores_log.txt"
    stdout = _run(
        [
            "uv", "run", "python", str(evaluator), str(graph_path), str(plan_path),
            "--config", str(config_path), "--output", str(result_path),
            "--trace-output", str(trace_path), "--log-output", str(log_path),
        ],
        timeout_seconds,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "result_path": str(result_path),
        "trace_path": str(trace_path),
        "log_path": str(log_path),
        "metrics": _metrics(result),
        "stdout": stdout,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：分两阶段跑 stub → 评测器，最后落盘 batch.json。"""
    parser = argparse.ArgumentParser(description="批量运行 artifacts/code 的多核 stub 评测")
    parser.add_argument("--cases-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "artifacts" / "data")
    parser.add_argument("-o", "--output-dir", type=Path,
                        default=Path("results/stub_multicore_cases"))
    parser.add_argument("--cores", nargs="+", type=int, default=list(CORE_COUNTS),
                        help="生成 stub plan 的核数；默认 1 2 3 4 5")
    parser.add_argument("--q", nargs="+", choices=QUESTIONS, default=list(QUESTIONS),
                        help="要运行的官方评估题目；默认 q1 q2 q3")
    parser.add_argument("--seed", type=int, default=0,
                        help="stub 的基础随机种子；实际 case_n 使用 seed+n")
    parser.add_argument("--min-subgraph-size", type=int, default=50)
    parser.add_argument("--max-subgraph-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help="评测并发数；默认读取环境变量 WORKERS（默认 4）")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS,
                        help="每次 stub 或评估器调用的超时秒数")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.timeout < 1:
        parser.error("--workers and --timeout must be positive")
    if any(core < 1 for core in args.cores):
        parser.error("--cores values must be positive")
    if args.min_subgraph_size < 1 or args.max_subgraph_size < args.min_subgraph_size:
        parser.error("invalid subgraph size range")

    graph_paths = discover_cases(args.cases_dir)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cores = tuple(sorted(set(args.cores)))
    questions = tuple(args.q)
    # Phase 1: 先用线程池跑完所有 stub plan。把它单独排空再开评测，是为了
    # 即使后续评测失败，所有输入方案也已落盘可供排查。
    states: list[dict[str, Any]] = [
        {
            "case": graph_path.stem,
            "graph": str(graph_path.resolve()),
            "config": str((graph_path.parent / "config.txt").resolve()),
            "runs": {},
            "errors": [],
        }
        for graph_path in graph_paths
    ]
    planning_jobs = [
        (index, graph_path, num_cores)
        for index, graph_path in enumerate(graph_paths)
        for num_cores in cores
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                generate_plan, graph_path, output_dir / graph_path.stem,
                num_cores, args.seed, args.min_subgraph_size,
                args.max_subgraph_size, args.timeout,
            ): (index, graph_path, num_cores)
            for index, graph_path, num_cores in planning_jobs
        }
        progress = tqdm(total=len(futures), desc="Generating stub plans", unit="plan", dynamic_ncols=True)
        try:
            for future in as_completed(futures):
                index, _, num_cores = futures[future]
                try:
                    states[index]["runs"][num_cores] = future.result()
                except Exception as exc:
                    states[index]["errors"].append(
                        f"stub {num_cores} cores: {type(exc).__name__}: {exc}"
                    )
                finally:
                    progress.update()
        finally:
            progress.close()

    # Phase 2: 仅评测 Phase 1 成功生成的方案。通常有 100 × 5 × 3 = 1500 个
    # 独立的官方评测进程，所以用线程池并发。
    evaluator_jobs = [
        (index, graph_path, num_cores, question)
        for index, graph_path in enumerate(graph_paths)
        for num_cores in cores
        if num_cores in states[index]["runs"]
        for question in questions
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                evaluate_plan, graph_path,
                Path(states[index]["runs"][num_cores]["plan"]),
                output_dir / graph_path.stem, num_cores, question, args.timeout,
            ): (index, num_cores, question)
            for index, graph_path, num_cores, question in evaluator_jobs
        }
        progress = tqdm(total=len(futures), desc="Evaluating stub plans", unit="run", dynamic_ncols=True)
        try:
            for future in as_completed(futures):
                index, num_cores, question = futures[future]
                try:
                    states[index]["runs"][num_cores]["problems"][question] = future.result()
                except Exception as exc:
                    states[index]["errors"].append(
                        f"{question}, {num_cores} cores: {type(exc).__name__}: {exc}"
                    )
                finally:
                    progress.update()
        finally:
            progress.close()

    # 汇总每 case 一个 aggregate.json，并按 evaluate_multicore 的口径计算 speedup。
    cases: list[dict[str, Any]] = []
    for state in states:
        runs = [state["runs"][num_cores] for num_cores in cores if num_cores in state["runs"]]
        # 与 evaluate_multicore.py 保持一致：只要有 1 核结果，就用它做 speedup
        # 基线。
        baseline: dict[str, int | float | None] = {}
        for question in questions:
            metrics = state["runs"].get(1, {}).get("problems", {}).get(question, {}).get("metrics", {})
            baseline[question] = metrics.get("makespan")
        for run in runs:
            for question, evaluation in run["problems"].items():
                makespan = evaluation["metrics"].get("makespan")
                reference = baseline.get(question)
                evaluation["metrics"]["speedup"] = (
                    reference / makespan if reference is not None and makespan else None
                )
        aggregate = {
            "graph": state["graph"],
            "config": state["config"],
            "stub": {
                "base_seed": args.seed,
                "min_subgraph_size": args.min_subgraph_size,
                "max_subgraph_size": args.max_subgraph_size,
            },
            "core_counts": list(cores),
            "baseline_core_count": 1,
            "baseline_makespan": baseline,
            "questions": list(questions),
            "runs": runs,
            "errors": state["errors"],
        }
        aggregate_path = output_dir / state["case"] / "aggregate.json"
        aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        aggregate_path.write_text(
            json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        cases.append({
            "case": state["case"],
            "graph": state["graph"],
            "status": "failed" if state["errors"] else "ok",
            "aggregate": str(aggregate_path),
            "runs": len(runs),
            **({"errors": state["errors"]} if state["errors"] else {}),
        })

    batch = {
        "cases_dir": str(args.cases_dir.resolve()),
        "output_dir": str(output_dir),
        "successful": sum(item["status"] == "ok" for item in cases),
        "failed": sum(item["status"] == "failed" for item in cases),
        "workers": args.workers,
        "planning_jobs": len(planning_jobs),
        "evaluator_jobs": len(evaluator_jobs),
        "core_counts": list(cores),
        "questions": list(questions),
        "cases": cases,
    }
    (output_dir / "batch.json").write_text(
        json.dumps(batch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Completed: {batch['successful']} succeeded, {batch['failed']} failed; {output_dir / 'batch.json'}")
    return 1 if batch["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
