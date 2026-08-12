from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ACTION_KEYS = {
    "resident": "consumption_stance",
    "firm": "labor_stance",
    "government": "policy_stance",
}


def _resident_buckets(observation: dict[str, Any]) -> list[tuple[str, str]]:
    duration = int(observation.get("unemployment_duration", 0))
    shock_duration = int(
        observation.get("shock_unemployment_duration", 0)
    )
    income_gap = float(observation.get("income_gap_ratio", 0.0))
    cash_gap = float(observation.get("cash_gap_months", 0.0))
    if duration >= 6:
        duration_bucket = "6+"
    elif duration >= 3:
        duration_bucket = "3-5"
    elif duration >= 1:
        duration_bucket = "1-2"
    else:
        duration_bucket = "0"
    if shock_duration >= 6:
        shock_duration_bucket = "6+"
    elif shock_duration >= 3:
        shock_duration_bucket = "3-5"
    elif shock_duration >= 1:
        shock_duration_bucket = "1-2"
    else:
        shock_duration_bucket = "0"
    if income_gap <= -0.20:
        income_bucket = "severe"
    elif income_gap <= -0.08:
        income_bucket = "moderate"
    else:
        income_bucket = "stable"
    if cash_gap <= -1.0:
        cash_bucket = "severe"
    elif cash_gap <= -0.5:
        cash_bucket = "moderate"
    else:
        cash_bucket = "stable"
    buckets = [
        ("unemployment_duration", duration_bucket),
        ("shock_unemployment_duration", shock_duration_bucket),
        ("income_gap", income_bucket),
        ("cash_gap", cash_bucket),
    ]
    if bool(observation.get("trend_available", False)):
        consumption_trend = float(
            observation.get("real_consumption_change_3m", 0.0)
        )
        if consumption_trend <= -0.05:
            trend_bucket = "falling"
        elif consumption_trend < 0.0:
            trend_bucket = "soft_fall"
        else:
            trend_bucket = "stable_or_rising"
        buckets.append(("real_consumption_trend", trend_bucket))
    return buckets


def _firm_buckets(observation: dict[str, Any]) -> list[tuple[str, str]]:
    utilization_gap = float(observation.get("utilization_gap", 0.0))
    cash_ratio = float(observation.get("cash_ratio", 1.0))
    if utilization_gap <= -0.13:
        utilization_bucket = "weak"
    elif utilization_gap >= 0.07:
        utilization_bucket = "strong"
    else:
        utilization_bucket = "balanced"
    if cash_ratio < 0.25:
        cash_bucket = "distressed"
    elif cash_ratio < 0.75:
        cash_bucket = "thin"
    else:
        cash_bucket = "adequate"
    buckets = [
        ("utilization_gap", utilization_bucket),
        ("cash_ratio", cash_bucket),
    ]
    if bool(observation.get("trend_available", False)):
        sales_trend = float(observation.get("firm_sales_change_3m", 0.0))
        if sales_trend <= -0.05:
            trend_bucket = "falling"
        elif sales_trend < 0.0:
            trend_bucket = "soft_fall"
        else:
            trend_bucket = "stable_or_rising"
        buckets.append(("firm_sales_trend", trend_bucket))
    return buckets


def _government_buckets(observation: dict[str, Any]) -> list[tuple[str, str]]:
    unemployment_gap = float(observation.get("unemployment_gap", 0.0))
    debt_ratio = float(observation.get("debt_ratio", 0.0))
    if unemployment_gap > 0.038:
        unemployment_bucket = "high"
    elif unemployment_gap > 0.008:
        unemployment_bucket = "elevated"
    else:
        unemployment_bucket = "near_target"
    if debt_ratio > 0.50:
        debt_bucket = "high"
    elif debt_ratio > 0.30:
        debt_bucket = "moderate"
    else:
        debt_bucket = "low"
    buckets = [
        ("unemployment_gap", unemployment_bucket),
        ("debt_ratio", debt_bucket),
    ]
    if bool(observation.get("trend_available", False)):
        consumption_trend = float(
            observation.get("household_consumption_change_3m", 0.0)
        )
        if consumption_trend <= -0.05:
            trend_bucket = "falling"
        elif consumption_trend < 0.0:
            trend_bucket = "soft_fall"
        else:
            trend_bucket = "stable_or_rising"
        buckets.append(("household_consumption_trend", trend_bucket))
    return buckets


def _condition_buckets(
    role: str, observation: dict[str, Any]
) -> list[tuple[str, str]]:
    if role == "resident":
        return _resident_buckets(observation)
    if role == "firm":
        return _firm_buckets(observation)
    if role == "government":
        return _government_buckets(observation)
    return []


def _selected_rate(
    conditioned: dict[tuple[str, str, str], Counter[str]],
    *,
    role: str,
    dimension: str,
    bucket: str,
    selected_actions: set[str],
) -> tuple[float | None, int]:
    actions = conditioned.get((role, dimension, bucket), Counter())
    total = sum(actions.values())
    if total == 0:
        return None, 0
    selected = sum(actions[action] for action in selected_actions)
    return selected / total, total


def _monotonicity_check(
    conditioned: dict[tuple[str, str, str], Counter[str]],
    *,
    name: str,
    role: str,
    dimension: str,
    reference_bucket: str,
    stress_bucket: str,
    selected_actions: set[str],
) -> dict[str, Any]:
    reference_rate, reference_records = _selected_rate(
        conditioned,
        role=role,
        dimension=dimension,
        bucket=reference_bucket,
        selected_actions=selected_actions,
    )
    stress_rate, stress_records = _selected_rate(
        conditioned,
        role=role,
        dimension=dimension,
        bucket=stress_bucket,
        selected_actions=selected_actions,
    )
    available = reference_rate is not None and stress_rate is not None
    return {
        "name": name,
        "role": role,
        "dimension": dimension,
        "reference_bucket": reference_bucket,
        "stress_bucket": stress_bucket,
        "selected_actions": sorted(selected_actions),
        "reference_rate": reference_rate,
        "stress_rate": stress_rate,
        "reference_records": reference_records,
        "stress_records": stress_records,
        "available": available,
        "passes_direction_check": (
            bool(stress_rate >= reference_rate) if available else None
        ),
    }


def summarize_decision_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    by_role: Counter[str] = Counter()
    by_status: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    action_distributions: dict[str, Counter[str]] = defaultdict(Counter)
    conditioned: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    record_count = 0
    active_record_count = 0
    llm_active_roles: set[str] = set()

    for record in records:
        record_count += 1
        role = str(record.get("role", "unknown"))
        status = str(record.get("status", "unknown"))
        observation = record.get("observation")
        observation = observation if isinstance(observation, dict) else {}
        final_action = record.get("final_action")
        final_action = final_action if isinstance(final_action, dict) else {}
        action_key = ACTION_KEYS.get(role)
        action = (
            str(final_action.get(action_key, "<missing>"))
            if action_key
            else "<missing>"
        )
        by_role[role] += 1
        by_status[status] += 1
        if status in {"accepted", "fallback", "bounded"}:
            llm_active_roles.add(role)
        if status == "inactive":
            continue
        active_record_count += 1
        action_distributions[role][action] += 1
        reason = record.get("fallback_reason")
        if reason:
            fallback_reasons[str(reason)] += 1
        phase = "post_shock" if bool(record.get("shock_active", False)) else "pre_shock"
        conditioned[(role, "phase", phase)][action] += 1
        for dimension, bucket in _condition_buckets(role, observation):
            conditioned[(role, dimension, bucket)][action] += 1

    conditioned_rows: list[dict[str, Any]] = []
    for (role, dimension, bucket), actions in sorted(conditioned.items()):
        total = sum(actions.values())
        for action, count in sorted(actions.items()):
            conditioned_rows.append(
                {
                    "role": role,
                    "dimension": dimension,
                    "bucket": bucket,
                    "action": action,
                    "records": count,
                    "share": count / total if total else 0.0,
                    "bucket_records": total,
                }
            )

    checks = [
        _monotonicity_check(
            conditioned,
            name="resident_income_stress_response",
            role="resident",
            dimension="income_gap",
            reference_bucket="stable",
            stress_bucket="severe",
            selected_actions={"cautious", "defensive"},
        ),
        _monotonicity_check(
            conditioned,
            name="resident_persistent_unemployment_response",
            role="resident",
            dimension="shock_unemployment_duration",
            reference_bucket="0",
            stress_bucket="6+",
            selected_actions={"cautious", "defensive"},
        ),
        _monotonicity_check(
            conditioned,
            name="firm_weak_demand_response",
            role="firm",
            dimension="utilization_gap",
            reference_bucket="balanced",
            stress_bucket="weak",
            selected_actions={"aggressive"},
        ),
        _monotonicity_check(
            conditioned,
            name="government_unemployment_response",
            role="government",
            dimension="unemployment_gap",
            reference_bucket="near_target",
            stress_bucket="high",
            selected_actions={"balanced_support", "stabilize"},
        ),
        _monotonicity_check(
            conditioned,
            name="government_debt_guard_response",
            role="government",
            dimension="debt_ratio",
            reference_bucket="low",
            stress_bucket="high",
            selected_actions={"fiscal_guard"},
        ),
    ]
    for item in checks:
        item["in_scope"] = item["role"] in llm_active_roles
    scoped_checks = [item for item in checks if bool(item["in_scope"])]
    available_checks = [
        item for item in scoped_checks if bool(item["available"])
    ]
    failed_checks = [
        item["name"]
        for item in available_checks
        if not bool(item["passes_direction_check"])
    ]
    unavailable_checks = [
        item["name"] for item in scoped_checks if not bool(item["available"])
    ]
    single_action_roles = sorted(
        role
        for role, actions in action_distributions.items()
        if role in llm_active_roles
        and sum(actions.values()) > 0
        and len(actions) == 1
    )
    qualification_warnings: list[str] = []
    if unavailable_checks:
        qualification_warnings.append("monotonicity_coverage_incomplete")
    if failed_checks:
        qualification_warnings.append("monotonicity_direction_failure")
    if single_action_roles:
        qualification_warnings.append("single_action_role_detected")
    if not llm_active_roles:
        qualification_warnings.append("no_llm_role_observed")
    suitable_for_behavior_claims = bool(llm_active_roles) and not (
        unavailable_checks or failed_checks or single_action_roles
    )
    return {
        "schema_version": 4,
        "records": record_count,
        "active_records": active_record_count,
        "by_role": dict(sorted(by_role.items())),
        "by_status": dict(sorted(by_status.items())),
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "action_distributions": {
            role: dict(sorted(actions.items()))
            for role, actions in sorted(action_distributions.items())
        },
        "conditioned_actions": conditioned_rows,
        "monotonicity_checks": checks,
        "behavior_qualification": {
            "suitable_for_behavior_claims": suitable_for_behavior_claims,
            "qualified_roles": sorted(llm_active_roles),
            "excluded_non_llm_roles": sorted(
                set(action_distributions) - llm_active_roles
            ),
            "available_monotonicity_checks": len(available_checks),
            "total_monotonicity_checks": len(scoped_checks),
            "failed_checks": failed_checks,
            "unavailable_checks": unavailable_checks,
            "single_action_roles": single_action_roles,
            "warnings": qualification_warnings,
        },
    }


def summarize_decision_audit(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    if source.is_file():
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid decision audit JSON at {source}:{line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Decision audit record at {source}:{line_number} is not an object"
                )
            records.append(record)
    result = summarize_decision_records(records)
    result["source"] = str(source)
    return result


def write_behavior_audit(
    decision_audit_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    result = summarize_decision_audit(decision_audit_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "decision_behavior_summary.json"
    csv_path = output / "decision_behavior_summary.csv"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fieldnames = (
        "role",
        "dimension",
        "bucket",
        "action",
        "records",
        "share",
        "bucket_records",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["conditioned_actions"])
    result["artifacts"] = {
        "json": json_path.name,
        "csv": csv_path.name,
    }
    return result
