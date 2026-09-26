# ============================================================
# 人工智能工具信息 | AI Tool Information
#   工具名称 (Tool Name)        : GLM-5.2
#   版本/型号 (Version/Model)   : GLM-5.2
#   开发机构/公司 (Developer)    : 智谱AI (Zhipu AI / zai-org)
#   版本颁布日期 (Release Date) : 2026-06-13
#   声明：本程序及代码是在人工智能工具辅助下完成的
# ============================================================
"""Q1 实验性 oracle 引导的种群搜索。

本模块刻意与 :mod:`q1_algorithm` 分开。它复用 Q1 的合法分区，只搜索“分区到
核”的分配，候选方案的适应度用官方 Q1 评测器（而非代理模型）测得。常规 Q1
入口保持不变。

搜索刻意保持小规模且确定。它适合做实验和回归对比，不会影响生产算法的可预
测运行时。
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
    """从环境变量读取整数，超界则夹到 [minimum, maximum]。

    所有搜索超参数都通过 ``_int_env`` 暴露，便于离线实验时按 case 调整，而
    不需要修改源码。``default`` 同时是回归测试使用的固定值。
    """
    try:
        value = int(os.environ.get(name, default))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _official_evaluate(graph: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """用仓库内置的官方 Q1 评测器评估单个候选方案。"""
    repo_root = Path(__file__).resolve().parents[2]
    evaluator = repo_root / "artifacts" / "code" / "multicore_cut_evaluate_problem_1.py"
    config = Path(os.environ.get("SUBGRAPH_SEARCH_CONFIG", ""))
    if not config.is_file():
        config = repo_root / "artifacts" / "excases" / "config.txt"
    timeout = _int_env("SUBGRAPH_SEARCH_ORACLE_TIMEOUT", 120, 1, 600)

    # 执行环境的 /tmp 可能是空间受限的 tmpfs。把 oracle 临时文件放到仓库旁边；
    # TemporaryDirectory 仍会在评估完成后立刻清理每个候选的文件。
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
    """把一个“分区→核”分配转换成可送评测器的方案 JSON。

    核内顺序按 ``topo_rank`` 排序，确保同核内不会出现后继在前驱之前的情况。
    """
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
    """返回分区 DAG 的真实拓扑序对应的位次。

    一个分区可能包含拓扑上不连续的多个算子位置。如果按“分区内第一个算子
    出现的顺序”对分区编号，可能把一条靠后的跨分区边反向，从而产生非法的同
    核 Task 顺序。因此这里跑一次真正的分区拓扑排序作为一个安全网。
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

    # “最早出现位置”作为拓扑排序的同秩 tie-break：保证在不引入环的前提下，
    # 分区序尽量贴合用户原始的算子序。
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
    """从 oracle 结果中提取目标：先比 makespan，平手再比 cross-task 字节数。"""
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
    """用小规模“官方 oracle 引导”的演化搜索构建 Q1 方案。

    搜索流程：
        1. 跑一次 Q1 基线，得到分区 + 初始分配；
        2. 计算分区 DAG 的真实拓扑序，作为同核内顺序的安全网；
        3. 用固定种子的 RNG 生成初始化种群（含基线个体 + 多个抖动个体）；
        4. 迭代 ``generations`` 轮：精英选择 → 均匀交叉 → 单点变异；
        5. 输出适应度最高的方案，并把搜索元数据写到 diagnostics。
    整个过程对 ``num_cores`` 和节点数确定性，便于回归复现。
    """
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
    # 把基线方案还原成一个 assignment tuple：每个分区在哪个核。
    baseline_assignment = [0] * len(subgraph_ids)
    for core, schedule in enumerate(baseline.plan["core_schedules"]):
        for subgraph_id in schedule:
            baseline_assignment[int(subgraph_id)] = core
    baseline_tuple = tuple(baseline_assignment)

    # 种群规模与代数都通过环境变量夹到合理区间：太小 (2,1) 没有搜索能力，
    # 太大 (32,20) 会让 oracle 调用成本失控。
    population_size = _int_env("SUBGRAPH_SEARCH_POPULATION", 8, 2, 32)
    generations = _int_env("SUBGRAPH_SEARCH_GENERATIONS", 3, 1, 20)
    # 种子固定且依赖 (num_cores, 节点数)，保证同一张图可复现。
    seed = 1729 + 31 * num_cores + len(features.topo_order)
    rng = random.Random(seed)
    population: list[tuple[int, ...]] = [baseline_tuple]
    # 基线之外的初始个体：随机改动 1~4 个分区所属核，对基线做局部抖动。
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
        # 1) 精英：取前 1/4 直接保留，确保每代都不退化。
        ranked = sorted(population, key=evaluate)
        elites = ranked[: max(1, population_size // 4)]
        next_population = list(elites)
        # 2) 繁殖：父本从精英中选，母本从前一半中选，单点交叉后 0.8 概率变异。
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
