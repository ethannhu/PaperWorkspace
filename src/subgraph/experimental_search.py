"""Experimental oracle-guided population search for Q1.

This module is deliberately separate from :mod:`q1_algorithm`.  It reuses
Q1's legal partitioning, then searches only the assignment of those
partitions to cores.  Candidate fitness is measured by the official Q1
evaluator, not by a proxy model.  The normal Q1 entry point is untouched.

The search is intentionally small and deterministic.  It is useful for
experiments and regression comparisons, while the production algorithm keeps
its predictable runtime.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .algorithm_common import GraphFeatures, analyze_graph
from .interfaces import AlgorithmResult
from .q1_algorithm import build_plan as build_q1_plan


def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _official_evaluate(graph: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one candidate through the checked-in official simulator."""
    repo_root = Path(__file__).resolve().parents[2]
    evaluator = repo_root / "artifacts" / "code" / "multicore_cut_evaluate_problem_1.py"
    config = Path(os.environ.get("SUBGRAPH_SEARCH_CONFIG", ""))
    if not config.is_file():
        config = repo_root / "artifacts" / "excases" / "config.txt"
    timeout = _int_env("SUBGRAPH_SEARCH_ORACLE_TIMEOUT", 120, 1, 600)

    # The execution environment may have a small, shared /tmp tmpfs.  Keep
    # oracle scratch files beside the repository instead; TemporaryDirectory
    # still removes every candidate's files immediately after evaluation.
    scratch_root = repo_root / ".oracle-tmp"
    scratch_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="subgraph-oracle-", dir=scratch_root) as directory:
        root = Path(directory)
        graph_path = root / "graph.json"
        plan_path = root / "plan.json"
        output_path = root / "result.json"
        graph_path.write_text(json.dumps(graph), encoding="utf-8")
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        command = [
            sys.executable,
            str(evaluator),
            str(graph_path),
            str(plan_path),
            "--config",
            str(config),
            "-o",
            str(output_path),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode != 0:
            details = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"official Q1 oracle failed: {details}")
        return json.loads(output_path.read_text(encoding="utf-8"))


def _candidate_plan(
    node_to_subgraph: dict[str, Any],
    assignment: tuple[int, ...],
    topo_rank: dict[int, int],
    num_cores: int,
) -> dict[str, Any]:
    by_core: list[list[int]] = [[] for _ in range(num_cores)]
    for subgraph_id, core in enumerate(assignment):
        by_core[core].append(subgraph_id)
    for schedule in by_core:
        schedule.sort(key=lambda subgraph_id: topo_rank[subgraph_id])
    return {
        "node_to_subgraph": node_to_subgraph,
        "core_schedules": by_core,
    }


def _partition_topological_rank(
    features: GraphFeatures,
    node_to_subgraph: dict[str, Any],
    subgraph_ids: list[int],
) -> dict[int, int]:
    """Return a true topological order of the partition DAG.

    A partition can contain non-contiguous operation positions.  Ordering
    partitions by their first operation therefore can reverse a later
    cross-partition edge and produce an invalid same-core Task order.
    """
    node_owner = {int(node_id): int(partition_id)
                  for node_id, partition_id in node_to_subgraph.items()}
    predecessors = {partition_id: set() for partition_id in subgraph_ids}
    successors = {partition_id: set() for partition_id in subgraph_ids}
    for source in features.topo_order:
        source_partition = node_owner[source]
        for target in features.succs[source]:
            target_partition = node_owner[target]
            if source_partition != target_partition:
                successors[source_partition].add(target_partition)
                predecessors[target_partition].add(source_partition)

    topo_position = {
        node_id: index for index, node_id in enumerate(features.topo_order)
    }
    earliest = {
        partition_id: min(
            topo_position[node_id]
            for node_id, owner in node_owner.items()
            if owner == partition_id
        )
        for partition_id in subgraph_ids
    }
    remaining = {partition_id: len(predecessors[partition_id])
                 for partition_id in subgraph_ids}
    ready = sorted(
        (partition_id for partition_id in subgraph_ids
         if remaining[partition_id] == 0),
        key=lambda partition_id: (earliest[partition_id], partition_id),
    )
    ordered: list[int] = []
    while ready:
        partition_id = ready.pop(0)
        ordered.append(partition_id)
        for successor in sorted(successors[partition_id]):
            remaining[successor] -= 1
            if remaining[successor] == 0:
                ready.append(successor)
                ready.sort(key=lambda item: (earliest[item], item))
    if len(ordered) != len(subgraph_ids):
        raise ValueError("Q1 partition graph contains a cycle")
    return {partition_id: index for index, partition_id in enumerate(ordered)}


def _score(result: dict[str, Any]) -> tuple[float, float]:
    makespan = float(result.get("makespan", float("inf")))
    movement = result.get("data_movement_bytes", float("inf"))
    if isinstance(movement, dict):
        traffic = float(movement.get("scheduled_copy_bytes", float("inf")))
    else:
        traffic = float(movement)
    return makespan, traffic


def build_plan(
    graph: dict[str, Any],
    num_cores: int = 4,
    features: GraphFeatures | None = None,
) -> AlgorithmResult:
    """Build a Q1 plan with a small official-oracle evolutionary search."""
    if num_cores < 1:
        raise ValueError("num_cores must be positive")
    if features is None:
        features = analyze_graph(graph)

    baseline = build_q1_plan(graph, num_cores=num_cores, features=features)
    node_to_subgraph = dict(baseline.plan["node_to_subgraph"])
    subgraph_ids = sorted({int(value) for value in node_to_subgraph.values()})
    if not subgraph_ids:
        return baseline
    if subgraph_ids != list(range(len(subgraph_ids))):
        raise ValueError("Q1 produced non-contiguous subgraph ids")

    topo_rank = _partition_topological_rank(
        features, node_to_subgraph, subgraph_ids
    )
    baseline_assignment = [0] * len(subgraph_ids)
    for core, schedule in enumerate(baseline.plan["core_schedules"]):
        for subgraph_id in schedule:
            baseline_assignment[int(subgraph_id)] = core
    baseline_tuple = tuple(baseline_assignment)

    population_size = _int_env("SUBGRAPH_SEARCH_POPULATION", 8, 2, 32)
    generations = _int_env("SUBGRAPH_SEARCH_GENERATIONS", 3, 1, 20)
    seed = 1729 + 31 * num_cores + len(features.topo_order)
    rng = random.Random(seed)
    population: list[tuple[int, ...]] = [baseline_tuple]
    for _ in range(population_size - 1):
        individual = list(baseline_tuple)
        moves = 1 + rng.randrange(max(1, min(4, len(individual))))
        for _ in range(moves):
            individual[rng.randrange(len(individual))] = rng.randrange(num_cores)
        population.append(tuple(individual))

    cache: dict[tuple[int, ...], tuple[tuple[float, float], dict[str, Any]]] = {}
    oracle_calls = 0

    def evaluate(individual: tuple[int, ...]) -> tuple[float, float]:
        nonlocal oracle_calls
        if individual not in cache:
            plan = _candidate_plan(node_to_subgraph, individual, topo_rank, num_cores)
            result = _official_evaluate(graph, plan)
            cache[individual] = (_score(result), plan)
            oracle_calls += 1
        return cache[individual][0]

    for _ in range(generations):
        ranked = sorted(population, key=evaluate)
        elites = ranked[: max(1, population_size // 4)]
        next_population = list(elites)
        while len(next_population) < population_size:
            left = rng.choice(elites)
            right = rng.choice(ranked[: max(2, population_size // 2)])
            cut = rng.randrange(1, len(left)) if len(left) > 1 else 0
            child = list(left[:cut] + right[cut:])
            if rng.random() < 0.8:
                child[rng.randrange(len(child))] = rng.randrange(num_cores)
            next_population.append(tuple(child))
        population = next_population

    best = min(population, key=evaluate)
    best_score, best_plan = cache[best]
    diagnostics = dict(baseline.diagnostics)
    diagnostics["experimental_search"] = {
        "enabled": True,
        "population": population_size,
        "generations": generations,
        "oracle": "artifacts/code/multicore_cut_evaluate_problem_1.py",
        "oracle_calls": oracle_calls,
        "baseline_score": cache[baseline_tuple][0],
        "best_score": best_score,
        "improved": best_score < cache[baseline_tuple][0],
    }
    return AlgorithmResult(plan=best_plan, diagnostics=diagnostics)


__all__ = ["build_plan"]
