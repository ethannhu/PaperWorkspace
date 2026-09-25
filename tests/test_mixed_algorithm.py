from __future__ import annotations

import json
import unittest
from pathlib import Path

from subgraph.demo_framework import build_plan
from subgraph.graph_patterns import GraphPattern, classify_graph


ROOT = Path(__file__).resolve().parents[1]


class MixedAlgorithmTests(unittest.TestCase):
    def test_all_mixed_cases_build_valid_plans(self) -> None:
        mixed = []
        for path in sorted((ROOT / "artifacts/data").glob("case_*.json")):
            graph = json.loads(path.read_text(encoding="utf-8"))
            if classify_graph(graph).pattern == GraphPattern.MIXED_MLP_REDUCE:
                mixed.append((path, graph))

        self.assertEqual(len(mixed), 22)
        for path, graph in mixed:
            result = build_plan(graph, num_cores=4, scenario="q2")
            plan = result.plan
            diagnostics = result.diagnostics["algorithm"]
            self.assertEqual(
                diagnostics["strategy"], "mixed_semantic_fusion_critical_path", path.name
            )
            self.assertEqual(len(plan["core_schedules"]), 4)
            scheduled = [pid for core in plan["core_schedules"] for pid in core]
            self.assertEqual(len(scheduled), len(set(scheduled)), path.name)
            self.assertEqual(
                set(plan["node_to_subgraph"].values()), set(scheduled), path.name
            )


if __name__ == "__main__":
    unittest.main()
