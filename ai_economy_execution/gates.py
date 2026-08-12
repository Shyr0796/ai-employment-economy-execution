from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any

from .models import EconomyState


ACCOUNTING_IDENTITIES = {
    "sales_identity_error": "firm_sales",
    "wage_identity_error": "gross_wage_bill",
    "tax_identity_error": "government_tax_revenue",
}
WARMUP_FLOW_METRICS = ("real_consumption", "aggregate_price", "firm_sales")
COUNTERFACTUAL_FLOW_METRICS = (
    "household_consumption",
    "real_consumption",
    "aggregate_price",
    "firm_sales",
)


def gate_thresholds(config: dict[str, Any]) -> dict[str, float]:
    configured = config.get("run_gate", {})
    return {
        "accounting_absolute_tolerance": float(configured.get("accounting_absolute_tolerance", 1e-8)),
        "accounting_relative_tolerance": float(configured.get("accounting_relative_tolerance", 1e-12)),
        "warmup_flow_max_relative_drift": float(configured.get("warmup_flow_max_relative_drift", 0.005)),
        "warmup_employment_max_gap": float(configured.get("warmup_employment_max_gap", 0.005)),
        "counterfactual_flow_max_relative_gap": float(
            configured.get("counterfactual_flow_max_relative_gap", 0.01)
        ),
        "counterfactual_employment_max_gap": float(configured.get("counterfactual_employment_max_gap", 0.005)),
        "systematic_pretrend_flow_max_relative_gap": float(
            configured.get("systematic_pretrend_flow_max_relative_gap", 0.005)
        ),
        "systematic_pretrend_employment_max_gap": float(
            configured.get("systematic_pretrend_employment_max_gap", 0.005)
        ),
    }


def initial_state_fingerprint(state: EconomyState) -> str:
    """Hash the initialized economic state while excluding the scenario label and empty runtime buffers."""
    payload = state.to_dict()
    payload.pop("scenario", None)
    payload.pop("intents", None)
    payload.pop("history", None)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _relative_gap(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1e-12)


def audit_history(
    history: list[dict[str, Any]],
    *,
    population: int,
    seed: int,
    scenario: str,
    warmup_months: int,
    thresholds: dict[str, float],
    equilibrium_reference: dict[str, float] | None = None,
) -> dict[str, Any]:
    warmup = [row for row in history if int(row["month"]) <= warmup_months]
    tail = warmup[-min(12, len(warmup)) :]
    employment_allowance = max(thresholds["warmup_employment_max_gap"], 1.0 / population)
    reference = equilibrium_reference or (warmup[0] if warmup else {})
    employment_reference = float(reference.get("employment_rate", math.nan))
    employment_drift = max(
        (
            abs(float(row["employment_rate"]) - employment_reference)
            for row in warmup
        ),
        default=math.inf,
    )
    tail_employment_drift = (
        abs(float(tail[-1]["employment_rate"]) - float(tail[0]["employment_rate"]))
        if len(tail) >= 2
        else math.inf
    )
    flow_drifts = {
        metric: max(
            (
                _relative_gap(float(row[metric]), float(reference.get(metric, math.nan)))
                for row in warmup
            ),
            default=math.inf,
        )
        for metric in WARMUP_FLOW_METRICS
    }
    tail_flow_drifts = {
        metric: _relative_gap(float(tail[-1][metric]), float(tail[0][metric]))
        if len(tail) >= 2
        else math.inf
        for metric in WARMUP_FLOW_METRICS
    }

    max_accounting_ratio = 0.0
    max_accounting_error = 0.0
    accounting_pass = True
    boundary_pass = bool(history) and len(warmup) == warmup_months
    for row in history:
        numeric_values = [float(value) for value in row.values() if isinstance(value, (int, float))]
        boundary_pass = boundary_pass and all(math.isfinite(value) for value in numeric_values)
        boundary_pass = boundary_pass and 0.0 <= float(row["employment_rate"]) <= 1.0
        boundary_pass = boundary_pass and float(row["aggregate_price"]) > 0.0
        for error_key, left_key in ACCOUNTING_IDENTITIES.items():
            error = abs(float(row[error_key]))
            left = float(row[left_key])
            right = left - float(row[error_key])
            allowed = max(
                thresholds["accounting_absolute_tolerance"],
                thresholds["accounting_relative_tolerance"] * max(abs(left), abs(right), 1.0),
            )
            max_accounting_error = max(max_accounting_error, error)
            max_accounting_ratio = max(max_accounting_ratio, error / allowed if allowed else math.inf)
            accounting_pass = accounting_pass and error <= allowed

    warmup_stability_pass = (
        employment_drift <= employment_allowance
        and all(
            value <= thresholds["warmup_flow_max_relative_drift"]
            for value in flow_drifts.values()
        )
    )
    return {
        "population": population,
        "seed": seed,
        "scenario": scenario,
        "warmup_observations": len(warmup),
        "warmup_employment_absolute_drift": employment_drift,
        "warmup_tail_employment_absolute_drift": tail_employment_drift,
        "warmup_employment_allowance": employment_allowance,
        **{f"warmup_{metric}_relative_drift": value for metric, value in flow_drifts.items()},
        **{
            f"warmup_tail_{metric}_relative_drift": value
            for metric, value in tail_flow_drifts.items()
        },
        "max_accounting_absolute_error": max_accounting_error,
        "max_accounting_tolerance_ratio": max_accounting_ratio,
        "accounting_pass": accounting_pass,
        "boundary_pass": boundary_pass,
        "warmup_stability_pass": warmup_stability_pass,
        "path_gate_pass": accounting_pass and boundary_pass and warmup_stability_pass,
    }


def audit_counterfactual_paths(
    histories: dict[str, list[dict[str, Any]]],
    fingerprints: dict[str, str],
    *,
    population: int,
    seed: int,
    warmup_months: int,
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    reference = histories["E0"]
    reference_by_month = {int(row["month"]): row for row in reference if int(row["month"]) <= warmup_months}
    employment_allowance = max(thresholds["counterfactual_employment_max_gap"], 1.0 / population)
    audits: list[dict[str, Any]] = []
    for scenario, history in histories.items():
        selected = [row for row in history if int(row["month"]) <= warmup_months]
        paired = [(row, reference_by_month.get(int(row["month"]))) for row in selected]
        complete = len(paired) == warmup_months and all(control is not None for _, control in paired)
        employment_gaps = [
            abs(float(row["employment_rate"]) - float(control["employment_rate"]))
            for row, control in paired
            if control is not None
        ]
        flow_gaps = {
            metric: [
                _relative_gap(float(row[metric]), float(control[metric]))
                for row, control in paired
                if control is not None
            ]
            for metric in COUNTERFACTUAL_FLOW_METRICS
        }
        endpoint = paired[-1] if paired else (None, None)
        endpoint_row, endpoint_control = endpoint
        initial_match = fingerprints[scenario] == fingerprints["E0"]
        max_employment_gap = max(employment_gaps, default=math.inf)
        max_flow_gaps = {metric: max(values, default=math.inf) for metric, values in flow_gaps.items()}
        path_pass = (
            complete
            and initial_match
            and max_employment_gap <= employment_allowance
            and all(
                value <= thresholds["counterfactual_flow_max_relative_gap"]
                for value in max_flow_gaps.values()
            )
        )
        row: dict[str, Any] = {
            "population": population,
            "seed": seed,
            "scenario": scenario,
            "control": "E0",
            "initial_state_match": initial_match,
            "warmup_path_complete": complete,
            "max_employment_rate_gap": max_employment_gap,
            "employment_gap_allowance": employment_allowance,
            "endpoint_employment_rate_signed_gap": (
                float(endpoint_row["employment_rate"]) - float(endpoint_control["employment_rate"])
                if endpoint_row is not None and endpoint_control is not None
                else math.inf
            ),
            "counterfactual_path_pass": path_pass,
        }
        for metric, value in max_flow_gaps.items():
            row[f"max_{metric}_relative_gap"] = value
            row[f"endpoint_{metric}_signed_relative_gap"] = (
                (float(endpoint_row[metric]) - float(endpoint_control[metric]))
                / max(abs(float(endpoint_control[metric])), 1e-12)
                if endpoint_row is not None and endpoint_control is not None
                else math.inf
            )
        audits.append(row)
    return audits


def summarize_run_gates(
    path_audits: list[dict[str, Any]],
    counterfactual_audits: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in counterfactual_audits:
        grouped[(int(row["population"]), str(row["scenario"]))].append(row)

    systematic: list[dict[str, Any]] = []
    for (population, scenario), rows in sorted(grouped.items()):
        employment_mean = sum(float(row["endpoint_employment_rate_signed_gap"]) for row in rows) / len(rows)
        flow_means = {
            metric: sum(float(row[f"endpoint_{metric}_signed_relative_gap"]) for row in rows) / len(rows)
            for metric in COUNTERFACTUAL_FLOW_METRICS
        }
        passed = (
            abs(employment_mean) <= thresholds["systematic_pretrend_employment_max_gap"]
            and all(
                abs(value) <= thresholds["systematic_pretrend_flow_max_relative_gap"]
                for value in flow_means.values()
            )
        )
        systematic.append({
            "population": population,
            "scenario": scenario,
            "seed_count": len(rows),
            "mean_endpoint_employment_rate_signed_gap": employment_mean,
            **{f"mean_endpoint_{metric}_signed_relative_gap": value for metric, value in flow_means.items()},
            "systematic_pretrend_pass": passed,
        })

    passed = (
        bool(path_audits)
        and all(bool(row["path_gate_pass"]) for row in path_audits)
        and all(bool(row["counterfactual_path_pass"]) for row in counterfactual_audits)
        and all(bool(row["systematic_pretrend_pass"]) for row in systematic)
    )
    return {
        "passed": passed,
        "thresholds": thresholds,
        "path_audits": path_audits,
        "counterfactual_audits": counterfactual_audits,
        "systematic_pretrend_audits": systematic,
    }
