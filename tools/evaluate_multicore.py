"""Run one partition/scheduling algorithm through the official Q1--Q3 evaluators.

The algorithm is supplied as ``module:callable`` and must return the standard
multicore plan dictionary.  The runner evaluates one-core baseline plus 2--5
cores, keeps every individual plan/evaluator result, writes one aggregate JSON
file, and saves a ``speedup.png`` plot.  When no graph is supplied, the runner
discovers and evaluates ``case_*.json`` files under ``artifacts/data`` and
displays case-level progress with tqdm; batch mode writes only each case's own
artifacts and no cross-case summary.

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
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from subgraph.interfaces import AlgorithmResult


PROBLEMS = (1, 2, 3)
CORE_COUNTS = (1, 2, 3, 4, 5)
DEFAULT_WORKERS = 4
DEFAULT_EVALUATOR_TIMEOUT_SECONDS = 600


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
    from subgraph.demo_framework import analyze_graph

    return {"features": analyze_graph(graph)}


def _run_evaluator(
    evaluator: Path,
    graph: Path,
    plan: Path,
    config: Path,
    output: Path,
    timeout_seconds: int | None = DEFAULT_EVALUATOR_TIMEOUT_SECONDS,
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


def _read_bandwidth(config_path: Path) -> int:
    active_section: str | None = None
    for line in config_path.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        if text.startswith("[") and text.endswith("]"):
            active_section = text[1:-1].strip().lower()
            continue
        if active_section == "bandwidth":
            parts = text.split()
            if len(parts) == 2 and parts[0] == "bandwidth":
                return int(parts[1])
    raise ValueError(f"{config_path}: missing [bandwidth] bandwidth setting")


def _copy_transfer_bytes(
    op: dict[str, Any],
    in_tids: dict[int, list[int]],
    out_tids: dict[int, list[int]],
    tensor_by_id: dict[int, dict[str, Any]],
) -> int:
    if op.get("op") == "COPY_IN":
        tids = out_tids.get(op["id"], [])
    elif op.get("op") == "COPY_OUT":
        tids = in_tids.get(op["id"], [])
    else:
        return 0
    return sum(int(tensor_by_id[tid].get("size", 0)) for tid in tids if tid in tensor_by_id)


def _graph_theoretical_metrics(
    graph: dict[str, Any],
    bandwidth: int,
    cores: tuple[int, ...],
) -> dict[str, Any]:
    """Compute input-only lower bounds and graph difficulty features.

    These values are intentionally independent of any algorithm output.  They
    are lower bounds or risk indicators, not a replacement for simulator
    makespan because the input graph contains no partition/core schedule.
    """
    ops = {int(item["id"]): item for item in graph.get("ops", [])}
    tensor_by_id = {int(item["id"]): item for item in graph.get("tensors", [])}
    in_edges: dict[int, list[int]] = defaultdict(list)
    out_edges: dict[int, list[int]] = defaultdict(list)
    for edge in graph.get("edges", []):
        source = int(edge["source"])
        target = int(edge["target"])
        out_edges[source].append(target)
        in_edges[target].append(source)

    in_tids: dict[int, list[int]] = {}
    out_tids: dict[int, list[int]] = {}
    for op_id in ops:
        in_tids[op_id] = [tid for tid in in_edges.get(op_id, []) if tid in tensor_by_id]
        out_tids[op_id] = [tid for tid in out_edges.get(op_id, []) if tid in tensor_by_id]

    succ: dict[int, set[int]] = defaultdict(set)
    pred: dict[int, set[int]] = defaultdict(set)
    produced_tensors = 0
    consumed_tensors = 0
    intermediate_tensor_bytes = 0
    fanout_tensor_bytes = 0
    for tensor_id, tensor in tensor_by_id.items():
        producers = [node for node in in_edges.get(tensor_id, []) if node in ops]
        consumers = [node for node in out_edges.get(tensor_id, []) if node in ops]
        if producers:
            produced_tensors += 1
        if consumers:
            consumed_tensors += 1
        if producers and consumers:
            size = int(tensor.get("size", 0))
            intermediate_tensor_bytes += size
            if len(consumers) > 1:
                fanout_tensor_bytes += size * (len(consumers) - 1)
        for source in producers:
            for target in consumers:
                if source != target:
                    succ[source].add(target)
                    pred[target].add(source)

    op_durations: dict[int, int] = {}
    copy_bytes_by_type = {"COPY_IN": 0, "COPY_OUT": 0}
    for op_id, op in ops.items():
        transfer_bytes = _copy_transfer_bytes(op, in_tids, out_tids, tensor_by_id)
        if op.get("op") in copy_bytes_by_type:
            copy_bytes_by_type[op["op"]] += transfer_bytes
            op_durations[op_id] = max(1, math.ceil(transfer_bytes / bandwidth)) if transfer_bytes else 0
        else:
            op_durations[op_id] = int(op.get("cycles", 0))

    indegree = {op_id: len(pred[op_id]) for op_id in ops}
    ready = deque(op_id for op_id, degree in indegree.items() if degree == 0)
    depth = {op_id: 0 for op_id in ready}
    longest_finish = {op_id: op_durations.get(op_id, 0) for op_id in ready}
    topo_count = 0
    while ready:
        op_id = ready.popleft()
        topo_count += 1
        finish = longest_finish.get(op_id, op_durations.get(op_id, 0))
        for nxt in succ[op_id]:
            depth[nxt] = max(depth.get(nxt, 0), depth.get(op_id, 0) + 1)
            longest_finish[nxt] = max(
                longest_finish.get(nxt, 0),
                finish + op_durations.get(nxt, 0),
            )
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
    if topo_count != len(ops):
        raise ValueError("input graph has a cycle in the folded op-DAG")

    layer_width = Counter(depth.values())
    op_counts = Counter(str(op.get("op")) for op in ops.values())
    pipe_work = Counter()
    for op_id, op in ops.items():
        pipe_work[str(op.get("pipe"))] += op_durations.get(op_id, 0)
    max_pipe_work = max(pipe_work.values(), default=0)
    total_duration = sum(op_durations.values())
    non_copy_cycles = sum(
        int(op.get("cycles", 0))
        for op in ops.values()
        if op.get("op") not in {"COPY_IN", "COPY_OUT"}
    )
    original_copy_bytes = copy_bytes_by_type["COPY_IN"] + copy_bytes_by_type["COPY_OUT"]
    ddr_time_lower_bound = original_copy_bytes / bandwidth if bandwidth else 0.0
    critical_path = max(longest_finish.values(), default=0)
    critical_path_non_copy = 0
    # The weighted critical path above includes estimated COPY durations.  This
    # companion value is useful when comparing purely computational structure.
    non_copy_duration = {
        op_id: (0 if ops[op_id].get("op") in {"COPY_IN", "COPY_OUT"} else int(ops[op_id].get("cycles", 0)))
        for op_id in ops
    }
    finish_non_copy: dict[int, int] = {}
    for op_id in sorted(ops, key=lambda item: depth.get(item, 0)):
        finish_non_copy[op_id] = max(
            [finish_non_copy[p] for p in pred[op_id]] or [0]
        ) + non_copy_duration[op_id]
    critical_path_non_copy = max(finish_non_copy.values(), default=0)

    lower_bounds: dict[str, dict[str, float]] = {}
    single_core_lower_bound = None
    for num_cores in sorted(set(cores)):
        work_bound = total_duration / num_cores if num_cores else 0.0
        pipe_bound = max_pipe_work / num_cores if num_cores else 0.0
        lower_bound = max(
            float(critical_path),
            work_bound,
            pipe_bound,
            ddr_time_lower_bound,
        )
        if num_cores == 1:
            single_core_lower_bound = lower_bound
        lower_bounds[str(num_cores)] = {
            "lower_bound_cycles": lower_bound,
            "work_bound_cycles": work_bound,
            "pipe_bound_cycles": pipe_bound,
            "critical_path_cycles": float(critical_path),
            "ddr_time_lower_bound_cycles": ddr_time_lower_bound,
        }
    if single_core_lower_bound is None:
        single_core_lower_bound = next(iter(lower_bounds.values()))["lower_bound_cycles"] if lower_bounds else 0.0
    for item in lower_bounds.values():
        bound = item["lower_bound_cycles"]
        item["input_speedup_upper_bound"] = (
            single_core_lower_bound / bound if bound else None
        )

    tensor_bytes_by_pos = Counter(str(tensor.get("pos")) for tensor in tensor_by_id.values())
    tensor_size_by_pos = Counter()
    for tensor in tensor_by_id.values():
        tensor_size_by_pos[str(tensor.get("pos"))] += int(tensor.get("size", 0))

    branch_nodes = sum(1 for op_id in ops if len(succ[op_id]) > 1)
    merge_nodes = sum(1 for op_id in ops if len(pred[op_id]) > 1)
    op_count = len(ops)
    return {
        "graph_size": {
            "op_count": op_count,
            "tensor_count": len(tensor_by_id),
            "edge_count": len(graph.get("edges", [])),
            "produced_tensor_count": produced_tensors,
            "consumed_tensor_count": consumed_tensors,
        },
        "op_counts": dict(sorted(op_counts.items())),
        "tensor_count_by_pos": dict(sorted(tensor_bytes_by_pos.items())),
        "tensor_bytes_by_pos": dict(sorted(tensor_size_by_pos.items())),
        "work": {
            "total_estimated_cycles": total_duration,
            "non_copy_compute_cycles": non_copy_cycles,
            "pipe_work_cycles": dict(sorted(pipe_work.items())),
            "max_pipe_work_cycles": max_pipe_work,
        },
        "copy": {
            "original_copy_bytes": original_copy_bytes,
            "copy_in_bytes": copy_bytes_by_type["COPY_IN"],
            "copy_out_bytes": copy_bytes_by_type["COPY_OUT"],
            "ddr_time_lower_bound_cycles": ddr_time_lower_bound,
        },
        "topology": {
            "critical_path_cycles": critical_path,
            "critical_path_non_copy_cycles": critical_path_non_copy,
            "dag_depth": max(depth.values(), default=0),
            "max_layer_width": max(layer_width.values(), default=0),
            "average_parallelism": total_duration / critical_path if critical_path else None,
            "branch_nodes": branch_nodes,
            "merge_nodes": merge_nodes,
            "branch_fraction": branch_nodes / op_count if op_count else 0.0,
            "merge_fraction": merge_nodes / op_count if op_count else 0.0,
        },
        "communication_risk": {
            "intermediate_tensor_bytes": intermediate_tensor_bytes,
            "fanout_tensor_extra_bytes": fanout_tensor_bytes,
            "fanout_to_original_copy_ratio": (
                fanout_tensor_bytes / original_copy_bytes if original_copy_bytes else None
            ),
        },
        "lower_bounds_by_core": lower_bounds,
    }


def evaluate(
    graph_path: Path,
    algorithm_spec: str,
    config_path: Path,
    output_dir: Path,
    cores: tuple[int, ...] = CORE_COUNTS,
    problems: tuple[int, ...] = PROBLEMS,
    evaluator_timeout: int | None = DEFAULT_EVALUATOR_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    graph_path = graph_path.resolve()
    config_path = config_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    simulator_output_dir = output_dir / "output"
    simulator_output_dir.mkdir(parents=True, exist_ok=True)
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    bandwidth = _read_bandwidth(config_path)
    theoretical_metrics = _graph_theoretical_metrics(graph, bandwidth, cores)
    algorithm = _load_callable(algorithm_spec)
    reusable_context = _prepare_reusable_context(algorithm, graph)
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
                reusable_context,
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
                evaluator_timeout,
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
        "theoretical_metrics": theoretical_metrics,
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
    algorithm_spec: str,
    output_dir: Path,
    config_path: Path | None = None,
    cores: tuple[int, ...] = CORE_COUNTS,
    problems: tuple[int, ...] = PROBLEMS,
    workers: int = DEFAULT_WORKERS,
    evaluator_timeout: int | None = DEFAULT_EVALUATOR_TIMEOUT_SECONDS,
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
        "algorithm": algorithm_spec,
        "successful": successful,
        "failed": len(cases) - successful,
        "workers": workers,
        "core_counts": list(cores),
        "problems": list(problems),
        "cases": cases,
    }


def plot_speedup(aggregate: dict[str, Any], output_dir: Path) -> Path:
    """Draw speedup trends together with input-derived theoretical bounds."""
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
    theoretical = aggregate.get("theoretical_metrics") or {}
    lower_bounds = theoretical.get("lower_bounds_by_core") or {}
    has_theoretical = bool(lower_bounds)

    if has_theoretical:
        fig, (ax, gap_ax) = plt.subplots(
            2,
            1,
            figsize=(3.5, 4.1),
            sharex=True,
            gridspec_kw={"height_ratios": [1.45, 1.0]},
            constrained_layout=False,
        )
    else:
        fig, ax = plt.subplots(figsize=(3.5, 2.7), constrained_layout=False)
        gap_ax = None
    for problem in ("q1", "q2", "q3"):
        problem_runs = [item for item in runs if problem in item["problems"]]
        problem_cores = [item["num_cores"] for item in problem_runs]
        values = [item["problems"][problem]["metrics"]["speedup"] for item in problem_runs]
        if not values:
            continue
        ax.plot(
            problem_cores,
            values,
            marker="o",
            markersize=4,
            linewidth=1.7,
            color=colors[problem],
            label=labels[problem],
        )
        if has_theoretical:
            baseline = aggregate.get("baseline_makespan", {}).get(problem)
            upper_values = []
            gap_values = []
            gap_cores = []
            for item in problem_runs:
                core_key = str(item["num_cores"])
                bound = lower_bounds.get(core_key, {}).get("lower_bound_cycles")
                makespan = item["problems"][problem]["metrics"].get("makespan")
                if not bound:
                    upper_values.append(None)
                    continue
                upper_values.append(baseline / bound if baseline else None)
                if makespan:
                    gap_cores.append(item["num_cores"])
                    gap_values.append(makespan / bound)
            if any(value is not None for value in upper_values):
                ax.plot(
                    problem_cores,
                    upper_values,
                    linewidth=1.1,
                    color=colors[problem],
                    alpha=0.45,
                    linestyle="--",
                    label=f"{labels[problem]} input bound",
                )
            if gap_ax is not None and gap_values:
                gap_ax.plot(
                    gap_cores,
                    gap_values,
                    marker="s",
                    markersize=3.2,
                    linewidth=1.25,
                    color=colors[problem],
                    label=labels[problem],
                )
    ax.axhline(1.0, color="#767676", linewidth=0.8, linestyle="--", zorder=0)
    if gap_ax is None:
        ax.set_xlabel("Number of cores")
    ax.set_ylabel("Speedup over 1 core")
    ax.set_xticks(core_counts)
    ax.set_xlim(min(core_counts) - 0.12, max(core_counts) + 0.12)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=2 if has_theoretical else 3, handlelength=1.8, columnspacing=1.0)
    if gap_ax is not None:
        gap_ax.axhline(1.0, color="#767676", linewidth=0.8, linestyle="--", zorder=0)
        gap_ax.set_xlabel("Number of cores")
        gap_ax.set_ylabel("Makespan / input lower bound")
        gap_ax.set_xticks(core_counts)
        gap_ax.set_xlim(min(core_counts) - 0.12, max(core_counts) + 0.12)
        gap_ax.set_ylim(bottom=0)
        gap_ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
        gap_ax.set_axisbelow(True)
    fig.tight_layout(pad=0.8)

    png_path = output_dir / "speedup.png"
    try:
        fig.savefig(png_path, dpi=600, bbox_inches="tight")
    finally:
        plt.close(fig)
    return png_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="运行 Q1-Q3 多核算法评测；省略 graph 时自动评测 cases 目录下全部 case"
    )
    parser.add_argument(
        "graph",
        type=Path,
        nargs="?",
        help="单个计算图 JSON；省略时进入批量模式",
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
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"批量模式并发线程数；默认 {DEFAULT_WORKERS}",
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
        "--problems",
        nargs="+",
        type=int,
        choices=PROBLEMS,
        default=list(PROBLEMS),
    )
    parser.add_argument(
        "--evaluator-timeout",
        type=int,
        default=DEFAULT_EVALUATOR_TIMEOUT_SECONDS,
        help=(
            "单次官方 evaluator 子进程超时秒数；"
            f"默认 {DEFAULT_EVALUATOR_TIMEOUT_SECONDS}，设为 0 表示不限制"
        ),
    )
    args = parser.parse_args(argv)
    if 1 not in args.cores:
        args.cores = [1, *args.cores]
    if args.evaluator_timeout < 0:
        raise ValueError("--evaluator-timeout must be non-negative")

    cores = tuple(sorted(set(args.cores)))
    problems = tuple(args.problems)
    evaluator_timeout = args.evaluator_timeout or None
    if args.graph is None:
        output_dir = args.output_dir or Path("results/multicore_cases")
        graph_paths = discover_cases(args.cases_dir)
        batch = evaluate_cases(
            graph_paths,
            args.algorithm,
            output_dir,
            args.config,
            cores,
            problems,
            workers=args.workers,
            evaluator_timeout=evaluator_timeout,
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
        evaluator_timeout,
    )
    figure_outputs = [str(plot_speedup(aggregate, output_dir))]
    print(json.dumps({
        "aggregate": str(output_dir / "aggregate.json"),
        "runs": len(aggregate["runs"]),
        "baseline_makespan": aggregate["baseline_makespan"],
        "figure_outputs": figure_outputs,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
