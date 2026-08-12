from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_economy_execution.configuration import load_config
from ai_economy_execution.experiments import parse_integer_set, run_numeric_experiment
from ai_economy_execution.metrics import (
    _demand_recovery_months,
    aggregate_comparisons,
    atkinson_index,
    quantile,
)
from ai_economy_execution.institutional_suite import _validate_suite
from ai_economy_execution.sensitivity import SENSITIVITY_SPECS, apply_sensitivity_value, run_sensitivity_analysis
from ai_economy_execution.strategy_experiment import run_strategy_experiment


class DistributionStatisticsTests(unittest.TestCase):
    def test_quantiles_and_atkinson(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(quantile(values, 0.25), 1.75)
        self.assertAlmostEqual(quantile(values, 0.75), 3.25)
        self.assertAlmostEqual(atkinson_index([5.0, 5.0, 5.0], 0.5), 0.0)
        self.assertGreater(atkinson_index([1.0, 2.0, 9.0], 1.0), 0.0)
        self.assertEqual(atkinson_index([0.0, 1.0], 1.5), 1.0)

    def test_bootstrap_summary_contains_median_and_iqr(self) -> None:
        rows = [
            {
                "population": 100,
                "seed": seed,
                "scenario": "E1",
                "control": "E0",
                "effect": value,
                "demand_recovery_months": seed,
                "ordinary_resident_benefit_pass": value > 0,
            }
            for seed, value in enumerate((1.0, 2.0, 3.0), start=1)
        ]
        result = aggregate_comparisons(rows, bootstrap_samples=200)
        self.assertEqual(result["effect"]["median"], 2.0)
        self.assertEqual(result["effect"]["q1"], 1.5)
        self.assertEqual(result["effect"]["q3"], 2.5)
        self.assertEqual(result["effect"]["iqr"], 1.0)
        self.assertIn("bootstrap_ci_low", result["effect"])


class DemandRecoveryTests(unittest.TestCase):
    @staticmethod
    def _pairs(ratios: list[float], months: list[int] | None = None):
        months = months or list(range(25, 25 + len(ratios)))
        return [
            (
                {"month": month, "household_consumption": 100.0 * ratio},
                {"month": month, "household_consumption": 100.0},
            )
            for month, ratio in zip(months, ratios)
        ]

    def test_never_breached_needs_no_recovery(self) -> None:
        pairs = self._pairs([1.0] * 8)
        self.assertEqual(_demand_recovery_months(pairs, shock_month=25), 0)

    def test_early_normal_window_does_not_mask_later_unrecovered_drop(self) -> None:
        pairs = self._pairs([1.0] * 6 + [0.98] * 8)
        self.assertIsNone(_demand_recovery_months(pairs, shock_month=25))

    def test_recovery_requires_six_consecutive_months_after_breach(self) -> None:
        pairs = self._pairs([1.0, 0.98, 0.995, 0.995, 0.98] + [0.995] * 6)
        self.assertEqual(_demand_recovery_months(pairs, shock_month=25), 5)

    def test_missing_calendar_month_breaks_recovery_window(self) -> None:
        pairs = self._pairs(
            [0.98] + [0.995] * 6,
            months=[25, 26, 27, 29, 30, 31, 32],
        )
        self.assertIsNone(_demand_recovery_months(pairs, shock_month=25))


class InstitutionalSuiteValidationTests(unittest.TestCase):
    @staticmethod
    def _histories() -> dict[str, list[dict[str, float | int | str]]]:
        histories = {}
        for scenario in ("E0", "E1", "E2", "E3", "E4"):
            post = {
                "month": 2,
                "scenario": scenario,
                "population": 10,
                "wage_employment": 7,
                "self_employment": 1,
                "unemployment_rate": 0.2,
            }
            if scenario == "E2":
                post.update(
                    cumulative_ai_attributable_layoffs_blocked=1,
                    cumulative_retention_wage_subsidy=10.0,
                    average_required_work_hours=150.0,
                )
            elif scenario == "E3":
                post.update(
                    cumulative_ai_levy_revenue=10.0,
                    cumulative_ai_levy_public_service_spending=7.0,
                    cumulative_ai_levy_public_investment=2.0,
                    government_ai_levy_fund_balance=1.0,
                )
            elif scenario == "E4":
                post.update(
                    cumulative_solo_entries=1,
                    solo_enterprise_sales=4.0,
                    solo_substitution_sales=1.0,
                    solo_b2b_sales=1.0,
                    solo_induced_demand_sales=1.0,
                    solo_external_sales=1.0,
                    cumulative_solo_induced_demand_sales=1.0,
                    cumulative_solo_external_sales=1.0,
                )
            histories[scenario] = [
                {
                    "month": 1,
                    "scenario": scenario,
                    "population": 10,
                    "economic_state_marker": 5.0,
                },
                post,
            ]
        return histories

    def test_scenario_label_is_not_a_pretreatment_state_mismatch(self) -> None:
        validation = _validate_suite(self._histories(), shock_month=2)
        self.assertTrue(validation["pass"])
        self.assertEqual(validation["common_pretreatment_mismatch_count"], 0)

    def test_economic_pretreatment_difference_is_detected(self) -> None:
        histories = self._histories()
        histories["E2"][0]["economic_state_marker"] = 5.1
        validation = _validate_suite(histories, shock_month=2)
        self.assertFalse(validation["checks"]["common_pretreatment_history"])
        self.assertEqual(validation["common_pretreatment_mismatch_count"], 1)


class AutomatedStudyTests(unittest.TestCase):
    def test_strategy_experiment_writes_main_matrix_and_competition_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "strategy"
            result = run_strategy_experiment(
                population=20,
                seeds=[1],
                months=25,
                output=output,
            )
            self.assertEqual(len(result["controls"]), 7)
            self.assertEqual(len(result["runs"]), 35)
            self.assertEqual(len(result["aggregate"]), 35)
            self.assertEqual(len(result["competition_paired"]), 20)
            self.assertEqual(len(result["competition_diagnostics"]), 20)
            self.assertEqual(len(result["policy_paired"]), 35)
            self.assertEqual(len(result["policy_diagnostics"]), 35)
            self.assertIn(
                "cumulative_employment_support_procurement_mean",
                result["aggregate"][0],
            )
            self.assertIn(
                "cumulative_productivity_dividend_procurement_mean",
                result["aggregate"][0],
            )
            self.assertFalse(result["design"]["export_sector_included"])
            self.assertTrue(result["design"]["matched_control_by_seed_and_firm_regime"])
            self.assertEqual(result["validation"]["max_abs_pretrend_gap"], 0.0)
            self.assertEqual(
                result["validation"]["support_exceeds_procurement_months"], 0
            )
            self.assertTrue(
                any(
                    path.endswith("configuration.py")
                    for path in result["source_hashes"]
                )
            )
            self.assertTrue((output / "strategy_matrix.csv").exists())
            self.assertTrue((output / "competition_paired.csv").exists())
            self.assertTrue((output / "competition_diagnostics.csv").exists())
            self.assertTrue((output / "policy_paired.csv").exists())
            self.assertTrue((output / "policy_diagnostics.csv").exists())
            self.assertTrue((output / "run_manifest.json").exists())
            self.assertTrue(
                (output / "paths" / "seed_1" / "augmentation" / "control_E0" / "metrics.csv").exists()
            )

    def test_integer_ranges_and_sensitivity_transformations(self) -> None:
        self.assertEqual(parse_integer_set("1-3,5,3"), [1, 2, 3, 5])
        config = load_config()
        ai_spec = next(item for item in SENSITIVITY_SPECS if item["id"] == "private_ai_productivity")
        modified = apply_sensitivity_value(config, ai_spec, 0.5)
        self.assertAlmostEqual(modified["firms"]["types"][0]["ai_target"], 1.10)
        self.assertAlmostEqual(config["firms"]["types"][0]["ai_target"], 1.20)

    def test_small_experiment_writes_complete_automatic_report(self) -> None:
        config = load_config()
        config["simulation"]["months"] = 30
        with tempfile.TemporaryDirectory(prefix="execution-report-test-") as temp_dir:
            output = Path(temp_dir)
            result = run_numeric_experiment(
                None,
                [20],
                [1, 2],
                output,
                baseline_config=config,
                write_paths=False,
                bootstrap_samples=100,
            )
            self.assertEqual(len(result["runs"]), 14)
            primary = result["aggregate"]["20"]["E1"]["bottom60_cumulative_real_consumption_gain"]
            self.assertIn("median", primary)
            self.assertIn("q1", primary)
            self.assertIn("bootstrap_ci_low", primary)
            inequality = result["aggregate"]["20"]["E1"]["tail_real_consumption_atkinson_1_0_delta"]
            self.assertIn("mean", inequality)
            self.assertIn("passed", result["run_gates"])
            self.assertTrue(all(
                row["initial_state_match"] for row in result["run_gates"]["counterfactual_audits"]
            ))
            for name in (
                "experiment_summary.json",
                "runs.csv",
                "comparisons.csv",
                "aggregate_statistics.csv",
                "run_gate_audit.csv",
                "research_report.md",
                "resolved_config.json",
            ):
                self.assertTrue((output / name).is_file(), name)

    def test_small_sensitivity_analysis_runs_all_boundaries(self) -> None:
        config = load_config()
        config["simulation"]["months"] = 26
        with tempfile.TemporaryDirectory(prefix="execution-sensitivity-test-") as temp_dir:
            output = Path(temp_dir)
            result = run_sensitivity_analysis(
                None,
                [20],
                [1],
                output,
                baseline_config=config,
                bootstrap_samples=20,
            )
            expected_variants = len(SENSITIVITY_SPECS) * 2
            self.assertEqual(len(list((output / "variant_configs").glob("*.json"))), expected_variants)
            self.assertGreater(len(result["effects"]), 0)
            self.assertTrue((output / "sensitivity_effects.csv").is_file())
            self.assertTrue((output / "sensitivity_report.md").is_file())


if __name__ == "__main__":
    unittest.main()
