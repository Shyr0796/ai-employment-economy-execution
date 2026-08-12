from __future__ import annotations

import unittest

import pandas as pd

from ai_economy_execution.reporting.generate_scientific_metric_atlas import (
    METRICS,
    SCENARIOS,
    full_y_axis_label,
    parse_args,
    safe_filename,
    transformed_values,
    validate_dataset,
)


def complete_test_frame() -> pd.DataFrame:
    metric_defaults = {spec.column: 0.0 for spec in METRICS}
    rows = []
    for regime_code in ("R1", "R2"):
        for scenario_code in SCENARIOS:
            for month in range(1, 26):
                rows.append(
                    {
                        **metric_defaults,
                        "regime_code": regime_code,
                        "scenario_code": scenario_code,
                        "seed": 1,
                        "population": 100,
                        "month": month,
                    }
                )
    return pd.DataFrame(rows)


class ScientificMetricAtlasTests(unittest.TestCase):
    def test_metric_catalog_and_filenames_are_unique(self) -> None:
        self.assertEqual(len(METRICS), 52)
        slugs = [spec.slug for spec in METRICS]
        filenames = [safe_filename("R2", spec) for spec in METRICS]
        self.assertEqual(len(slugs), len(set(slugs)))
        self.assertEqual(len(filenames), len(set(filenames)))

    def test_complete_matrix_passes_preflight(self) -> None:
        validate_dataset(complete_test_frame())

    def test_missing_scenario_is_rejected(self) -> None:
        frame = complete_test_frame()
        frame = frame[
            ~(
                (frame["regime_code"] == "R2")
                & (frame["scenario_code"] == "E6")
            )
        ]
        with self.assertRaisesRegex(ValueError, "scenario matrix mismatch"):
            validate_dataset(frame)

    def test_duplicate_month_key_is_rejected(self) -> None:
        frame = complete_test_frame()
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_dataset(frame)

    def test_validate_only_cli_can_target_one_regime(self) -> None:
        args = parse_args(["--validate-only", "--regimes", "R2"])
        self.assertTrue(args.validate_only)
        self.assertEqual(args.regimes, ["R2"])

    def test_tex_ready_cli_flag_is_explicit(self) -> None:
        args = parse_args(["--tex-ready"])
        self.assertTrue(args.tex_ready)

    def test_every_y_axis_label_identifies_its_metric(self) -> None:
        for spec in METRICS:
            with self.subTest(metric=spec.slug):
                self.assertTrue(full_y_axis_label(spec).startswith(spec.title))

    def test_public_service_axis_is_written_in_full(self) -> None:
        spec = next(item for item in METRICS if item.slug == "public_service_index")
        self.assertEqual(full_y_axis_label(spec), "Public-service index")

    def test_undefined_capital_ratio_sentinel_is_not_plotted(self) -> None:
        spec = next(item for item in METRICS if item.slug == "bank_capital_adequacy")
        frame = pd.DataFrame({spec.column: [999.0, -999.0, 0.12]})
        values = transformed_values(frame, spec)
        self.assertTrue(pd.isna(values.iloc[0]))
        self.assertTrue(pd.isna(values.iloc[1]))
        self.assertEqual(values.iloc[2], 12.0)


if __name__ == "__main__":
    unittest.main()
