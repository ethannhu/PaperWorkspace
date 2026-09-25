"""Regression coverage for the 20 representative graph-pattern cases."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from subgraph.graph_patterns import GraphPattern, GraphPatternFamily, classify_graph


ROOT = Path(__file__).resolve().parents[1]


class GraphPatternClassifierTests(unittest.TestCase):
    def test_representative_cases_cover_all_seven_patterns(self) -> None:
        expected = {
            "case_001": GraphPattern.SHALLOW_WIDE_COMPUTE_ACTIVATION,
            "case_003": GraphPattern.ATTENTION_NORMALIZE,
            "case_004": GraphPattern.GATED_SIGMOID_MLP,
            "case_006": GraphPattern.MIXED_MLP_REDUCE,
            "case_007": GraphPattern.GATED_SIGMOID_MLP,
            "case_010": GraphPattern.MIXED_MLP_REDUCE,
            "case_011": GraphPattern.WIDE_MATMUL_ADD,
            "case_016": GraphPattern.NARROW_DEEP_REDUCE_RELU_ADD,
            "case_020": GraphPattern.CNN_RESIDUAL,
            "case_024": GraphPattern.NARROW_DEEP_REDUCE_RELU_ADD,
            "case_036": GraphPattern.SHALLOW_WIDE_COMPUTE_ACTIVATION,
            "case_044": GraphPattern.CNN_RESIDUAL,
            "case_047": GraphPattern.ATTENTION_NORMALIZE,
            "case_052": GraphPattern.MIXED_MLP_REDUCE,
            "case_058": GraphPattern.WIDE_MATMUL_ADD,
            "case_062": GraphPattern.SHALLOW_WIDE_COMPUTE_ACTIVATION,
            "case_068": GraphPattern.ATTENTION_NORMALIZE,
            "case_086": GraphPattern.ATTENTION_NORMALIZE,
            "case_090": GraphPattern.CNN_RESIDUAL,
            "case_100": GraphPattern.MIXED_MLP_REDUCE,
        }
        observed = {}
        observed_families = set()
        for case, pattern in expected.items():
            graph = json.loads((ROOT / "artifacts/excases" / f"{case}.json").read_text())
            report = classify_graph(graph)
            observed[case] = report.pattern
            observed_families.add(report.family)
            self.assertEqual(pattern, observed[case], case)
        self.assertEqual(set(GraphPattern), set(observed.values()))
        self.assertEqual(set(GraphPatternFamily), observed_families)
