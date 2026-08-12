from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_economy_execution.cognitive_matrix_suite import (
    build_parser,
    build_plan,
    parse_regimes,
    parse_scenarios,
)


class CognitiveMatrixSuiteTests(unittest.TestCase):
    def test_range_parsers_preserve_registered_order(self) -> None:
        self.assertEqual(parse_regimes("R1-R3,R0"), ["R0", "R1", "R2", "R3"])
        self.assertEqual(parse_scenarios("E5,E0-E2"), ["E0", "E1", "E2", "E5"])
        with self.assertRaises(ValueError):
            parse_regimes("R4")

    def test_plan_contains_one_shared_equilibrium_and_all_matrix_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = build_parser().parse_args(
                [
                    "--population",
                    "100",
                    "--months",
                    "36",
                    "--seeds",
                    "1",
                    "--matrix-root",
                    str(Path(temporary) / "matrix"),
                ]
            )
            plan = build_plan(args)
        self.assertEqual(plan["counts"]["equilibria"], 1)
        self.assertEqual(plan["counts"]["cells"], 28)
        self.assertEqual(plan["counts"]["paid_cells"], 21)
        self.assertEqual(len(plan["tasks"]), 29)
        self.assertEqual(plan["tasks"][0]["kind"], "equilibrium")
        r1_e0 = next(
            task
            for task in plan["tasks"]
            if task["id"] == "S001:R1:E0"
        )
        self.assertTrue(r1_e0["paid"])
        self.assertIn(
            "--activate-cognitive-regime-from-checkpoint",
            r1_e0["command"],
        )
        self.assertIn("R1_government", r1_e0["output"])


if __name__ == "__main__":
    unittest.main()
