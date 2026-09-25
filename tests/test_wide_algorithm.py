from __future__ import annotations

import json
import unittest
from pathlib import Path

from subgraph.q2_algorithm import build_plan
from subgraph.graph_patterns import GraphPattern, classify_graph


ROOT = Path(__file__).resolve().parents[1]


class WideAlgorithmTests(unittest.TestCase):
    def test_wide_representatives_use_specialized_strategy(self) -> None:
        for case_id in ("001", "004", "007", "011", "036", "058", "062"):
            graph = json.loads(
                (ROOT / "artifacts/excases" / f"case_{case_id}.json").read_text(
                    encoding="utf-8"
                )
            )
            result = build_plan(graph, num_cores=4)
            self.assertEqual(
                classify_graph(graph).family.value,
                "wide",
                case_id,
            )
            self.assertEqual(
                result.diagnostics["algorithm"]["strategy"],
                "wide_communication_aware_coalescing",
                case_id,
            )
            plan = result.plan
            scheduled = [pid for core in plan["core_schedules"] for pid in core]
            self.assertEqual(len(plan["core_schedules"]), 4, case_id)
            self.assertEqual(len(scheduled), len(set(scheduled)), case_id)
            self.assertEqual(
                set(plan["node_to_subgraph"].values()), set(scheduled), case_id
            )

    def test_large_wide_graph_is_coalesced(self) -> None:
        graph = json.loads(
            (ROOT / "artifacts/excases/case_058.json").read_text(encoding="utf-8")
        )
        result = build_plan(graph, num_cores=4)
        algorithm = result.diagnostics["algorithm"]
        self.assertEqual(algorithm["motif"], "matmul_add_fan_in")
        self.assertLess(algorithm["partition_count"], algorithm["base_partition_count"])
        self.assertLessEqual(algorithm["max_ops"], 32)


if __name__ == "__main__":
    unittest.main()
