from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


INCOME_GROUPS = ("low", "lower_middle", "middle", "upper_middle", "high")
ATKINSON_EPSILONS = (0.5, 1.0, 1.5)


def quantile(values: list[float], probability: float) -> float:
    """Return a linearly interpolated sample quantile."""
    if not values:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be inside [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def atkinson_index(values: list[float], epsilon: float) -> float:
    """Compute the Atkinson inequality index for non-negative observations."""
    if not values:
        return 0.0
    observations = [float(value) for value in values]
    if any(value < 0.0 for value in observations):
        raise ValueError("Atkinson index is undefined for negative observations")
    arithmetic_mean = mean(observations)
    if arithmetic_mean == 0.0:
        return 0.0
    if epsilon < 0.0:
        raise ValueError("Atkinson epsilon must be non-negative")
    if epsilon == 1.0:
        if any(value == 0.0 for value in observations):
            return 1.0
        equally_distributed_equivalent = math.exp(mean([math.log(value) for value in observations]))
    else:
        power = 1.0 - epsilon
        if power < 0.0 and any(value == 0.0 for value in observations):
            return 1.0
        equally_distributed_equivalent = mean([value**power for value in observations]) ** (1.0 / power)
    return min(1.0, max(0.0, 1.0 - equally_distributed_equivalent / arithmetic_mean))


def _epsilon_label(epsilon: float) -> str:
    return str(epsilon).replace(".", "_")


def _resilience_observation(
    resident: Any,
    liquidity_vulnerability_months: float,
    cash_vulnerability_months: float,
    consumption_stress_ratio: float,
    unemployment_stress_months: int,
) -> dict[str, float | bool]:
    essential_spending = max(float(resident.minimum_consumption), 1e-9)
    cash_buffer_months = float(resident.cash) / essential_spending
    liquid_buffer_months = (
        float(resident.cash) + float(resident.deposits)
    ) / essential_spending
    cash_vulnerable = float(resident.cash) < float(resident.target_cash)
    essential_cash_shortfall = cash_buffer_months < cash_vulnerability_months
    liquidity_vulnerable = (
        liquid_buffer_months < liquidity_vulnerability_months
    )
    income_stress = float(resident.disposable_income) < essential_spending
    consumption_compressed = float(resident.real_consumption) < (
        consumption_stress_ratio * float(resident.baseline_consumption)
    )
    persistent_unemployment = bool(
        not resident.employed
        and resident.unemployment_duration >= unemployment_stress_months
    )
    shock_persistent_unemployment = bool(
        resident.shock_unemployed
        and not resident.employed
        and resident.shock_unemployment_duration >= unemployment_stress_months
    )
    economic_stress = bool(
        liquid_buffer_months < cash_vulnerability_months
        or income_stress
        or consumption_compressed
        or shock_persistent_unemployment
    )
    return {
        "cash_buffer_months": cash_buffer_months,
        "liquid_buffer_months": liquid_buffer_months,
        "cash_vulnerable": cash_vulnerable,
        "essential_cash_shortfall": essential_cash_shortfall,
        "liquidity_vulnerable": liquidity_vulnerable,
        "income_stress": income_stress,
        "consumption_compressed": consumption_compressed,
        "persistent_unemployment": persistent_unemployment,
        "shock_persistent_unemployment": shock_persistent_unemployment,
        "economic_stress": economic_stress,
    }


def _distribution_block(
    prefix: str,
    residents: list[Any],
    liquidity_vulnerability_months: float,
    cash_vulnerability_months: float,
    consumption_stress_ratio: float,
    unemployment_stress_months: int,
) -> dict[str, float]:
    if not residents:
        return {
            f"{prefix}_population": 0,
            f"{prefix}_employment_rate": 0.0,
            f"{prefix}_disposable_income": 0.0,
            f"{prefix}_mean_disposable_income": 0.0,
            f"{prefix}_real_consumption": 0.0,
            f"{prefix}_mean_real_consumption": 0.0,
            f"{prefix}_cash": 0.0,
            f"{prefix}_mean_cash": 0.0,
            f"{prefix}_deposits": 0.0,
            f"{prefix}_mean_deposits": 0.0,
            f"{prefix}_managed_fund_assets": 0.0,
            f"{prefix}_mean_managed_fund_assets": 0.0,
            f"{prefix}_financial_wealth": 0.0,
            f"{prefix}_mean_financial_wealth": 0.0,
            f"{prefix}_mean_target_cash": 0.0,
            f"{prefix}_mean_cash_buffer_months": 0.0,
            f"{prefix}_mean_liquid_buffer_months": 0.0,
            f"{prefix}_cash_vulnerable_rate": 0.0,
            f"{prefix}_essential_cash_shortfall_rate": 0.0,
            f"{prefix}_liquidity_vulnerable_rate": 0.0,
            f"{prefix}_income_stress_rate": 0.0,
            f"{prefix}_consumption_compression_rate": 0.0,
            f"{prefix}_persistent_unemployment_rate": 0.0,
            f"{prefix}_shock_persistent_unemployment_rate": 0.0,
            f"{prefix}_economic_stress_rate": 0.0,
        }
    count = len(residents)
    disposable_income = [float(item.disposable_income) for item in residents]
    real_consumption = [float(item.real_consumption) for item in residents]
    cash = [float(item.cash) for item in residents]
    deposits = [float(item.deposits) for item in residents]
    managed_fund_assets = [float(item.managed_fund_assets) for item in residents]
    financial_wealth = [
        cash_value + deposit + managed
        for cash_value, deposit, managed in zip(
            cash, deposits, managed_fund_assets
        )
    ]
    resilience = [
        _resilience_observation(
            item,
            liquidity_vulnerability_months,
            cash_vulnerability_months,
            consumption_stress_ratio,
            unemployment_stress_months,
        )
        for item in residents
    ]
    return {
        f"{prefix}_population": count,
        f"{prefix}_employment_rate": sum(bool(item.employed) for item in residents) / count,
        f"{prefix}_disposable_income": sum(disposable_income),
        f"{prefix}_mean_disposable_income": mean(disposable_income),
        f"{prefix}_real_consumption": sum(real_consumption),
        f"{prefix}_mean_real_consumption": mean(real_consumption),
        f"{prefix}_cash": sum(cash),
        f"{prefix}_mean_cash": mean(cash),
        f"{prefix}_deposits": sum(deposits),
        f"{prefix}_mean_deposits": mean(deposits),
        f"{prefix}_managed_fund_assets": sum(managed_fund_assets),
        f"{prefix}_mean_managed_fund_assets": mean(managed_fund_assets),
        f"{prefix}_financial_wealth": sum(financial_wealth),
        f"{prefix}_mean_financial_wealth": mean(financial_wealth),
        f"{prefix}_mean_target_cash": mean(
            [float(item.target_cash) for item in residents]
        ),
        f"{prefix}_mean_cash_buffer_months": mean(
            [float(item["cash_buffer_months"]) for item in resilience]
        ),
        f"{prefix}_mean_liquid_buffer_months": mean(
            [float(item["liquid_buffer_months"]) for item in resilience]
        ),
        f"{prefix}_cash_vulnerable_rate": sum(
            bool(item["cash_vulnerable"]) for item in resilience
        )
        / count,
        f"{prefix}_essential_cash_shortfall_rate": sum(
            bool(item["essential_cash_shortfall"]) for item in resilience
        )
        / count,
        f"{prefix}_liquidity_vulnerable_rate": sum(
            bool(item["liquidity_vulnerable"]) for item in resilience
        )
        / count,
        f"{prefix}_income_stress_rate": sum(
            bool(item["income_stress"]) for item in resilience
        )
        / count,
        f"{prefix}_consumption_compression_rate": sum(
            bool(item["consumption_compressed"]) for item in resilience
        )
        / count,
        f"{prefix}_persistent_unemployment_rate": sum(
            bool(item["persistent_unemployment"]) for item in resilience
        )
        / count,
        f"{prefix}_shock_persistent_unemployment_rate": sum(
            bool(item["shock_persistent_unemployment"]) for item in resilience
        )
        / count,
        f"{prefix}_economic_stress_rate": sum(
            bool(item["economic_stress"]) for item in resilience
        )
        / count,
    }


def resident_distribution_metrics(
    residents: list[Any],
    liquidity_vulnerability_months: float = 3.0,
    cash_vulnerability_months: float = 1.0,
    consumption_stress_ratio: float = 0.85,
    unemployment_stress_months: int = 6,
) -> dict[str, float]:
    """Create flat monthly distribution metrics suitable for JSON and CSV output."""
    if not residents:
        return {}
    result: dict[str, float] = {}
    disposable_income = [float(item.disposable_income) for item in residents]
    real_consumption = [float(item.real_consumption) for item in residents]
    cash = [float(item.cash) for item in residents]
    deposits = [float(item.deposits) for item in residents]
    managed_fund_assets = [float(item.managed_fund_assets) for item in residents]
    financial_wealth = [
        cash_value + deposit + managed
        for cash_value, deposit, managed in zip(
            cash, deposits, managed_fund_assets
        )
    ]
    resilience = [
        _resilience_observation(
            item,
            liquidity_vulnerability_months,
            cash_vulnerability_months,
            consumption_stress_ratio,
            unemployment_stress_months,
        )
        for item in residents
    ]
    for name, observations in (
        ("disposable_income", disposable_income),
        ("real_consumption", real_consumption),
        ("cash", cash),
        ("deposits", deposits),
        ("managed_fund_assets", managed_fund_assets),
        ("financial_wealth", financial_wealth),
    ):
        result[f"{name}_mean"] = mean(observations)
        result[f"{name}_median"] = median(observations)
        result[f"{name}_q1"] = quantile(observations, 0.25)
        result[f"{name}_q3"] = quantile(observations, 0.75)
        result[f"{name}_iqr"] = result[f"{name}_q3"] - result[f"{name}_q1"]
    result["disposable_income"] = sum(disposable_income)
    result["mean_target_cash"] = mean(
        [float(item.target_cash) for item in residents]
    )
    result["mean_cash_buffer_months"] = mean(
        [float(item["cash_buffer_months"]) for item in resilience]
    )
    result["mean_liquid_buffer_months"] = mean(
        [float(item["liquid_buffer_months"]) for item in resilience]
    )
    for key in (
        "cash_vulnerable",
        "essential_cash_shortfall",
        "liquidity_vulnerable",
        "income_stress",
        "consumption_compressed",
        "persistent_unemployment",
        "shock_persistent_unemployment",
        "economic_stress",
    ):
        output = {
            "cash_vulnerable": "cash_vulnerable_rate",
            "essential_cash_shortfall": "essential_cash_shortfall_rate",
            "liquidity_vulnerable": "liquidity_vulnerable_rate",
            "income_stress": "income_stress_rate",
            "consumption_compressed": "consumption_compression_rate",
            "persistent_unemployment": "persistent_unemployment_rate",
            "shock_persistent_unemployment": (
                "shock_persistent_unemployment_rate"
            ),
            "economic_stress": "economic_stress_rate",
        }[key]
        result[output] = sum(bool(item[key]) for item in resilience) / len(residents)
    for epsilon in ATKINSON_EPSILONS:
        label = _epsilon_label(epsilon)
        result[f"disposable_income_atkinson_{label}"] = atkinson_index(disposable_income, epsilon)
        result[f"real_consumption_atkinson_{label}"] = atkinson_index(real_consumption, epsilon)

    by_group = {group: [item for item in residents if item.income_group == group] for group in INCOME_GROUPS}
    for group, selected in by_group.items():
        result.update(
            _distribution_block(
                f"group_{group}",
                selected,
                liquidity_vulnerability_months,
                cash_vulnerability_months,
                consumption_stress_ratio,
                unemployment_stress_months,
            )
        )
    result.update(
        _distribution_block(
            "bottom60",
            [item for group in INCOME_GROUPS[:3] for item in by_group[group]],
            liquidity_vulnerability_months,
            cash_vulnerability_months,
            consumption_stress_ratio,
            unemployment_stress_months,
        )
    )
    result.update(
        _distribution_block(
            "bottom80",
            [item for group in INCOME_GROUPS[:4] for item in by_group[group]],
            liquidity_vulnerability_months,
            cash_vulnerability_months,
            consumption_stress_ratio,
            unemployment_stress_months,
        )
    )
    return result


def validate_metric(
    metric: dict[str, Any],
    *,
    absolute_tolerance: float = 1e-8,
    relative_tolerance: float = 1e-12,
) -> None:
    identities = {
        "sales_identity_error": "firm_sales",
        "wage_identity_error": "gross_wage_bill",
        "tax_identity_error": "government_tax_revenue",
        "bank_balance_sheet_error": "bank_total_funding",
    }
    for error_key, left_key in identities.items():
        if error_key not in metric:
            continue
        error = float(metric[error_key])
        left = float(metric[left_key])
        right = left - error
        scale = max(abs(left), abs(right), 1.0)
        tolerance = max(absolute_tolerance, relative_tolerance * scale)
        if abs(error) > tolerance:
            raise ValueError(
                f"Accounting identity failed: {error_key}={error}; "
                f"allowed={tolerance} (absolute={absolute_tolerance}, relative={relative_tolerance})"
            )
    if not 0.0 <= float(metric["employment_rate"]) <= 1.0:
        raise ValueError("employment_rate outside [0, 1]")
    if float(metric["aggregate_price"]) <= 0.0:
        raise ValueError("aggregate_price must be positive")


def write_history(history: list[dict[str, Any]], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    if history:
        with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)


def summarize(history: list[dict[str, Any]], shock_month: int = 25) -> dict[str, Any]:
    post = [row for row in history if int(row["month"]) >= shock_month]
    tail = post[-12:] if post else []
    result: dict[str, Any] = {
        "post_months": len(post),
        "cumulative_real_consumption": sum(float(row["real_consumption"]) for row in post),
        "bottom60_cumulative_real_consumption": sum(float(row["bottom60_real_consumption"]) for row in post),
        "bottom80_cumulative_real_consumption": sum(float(row["bottom80_real_consumption"]) for row in post),
        "cumulative_disposable_income": sum(float(row["disposable_income"]) for row in post),
        "cumulative_real_government_purchase": sum(
            float(row.get("government_real_purchase", 0.0)) for row in post
        ),
        "cumulative_real_government_procurement": sum(
            float(row.get("government_real_procurement", 0.0)) for row in post
        ),
        "cumulative_employment_support_procurement": sum(
            float(row.get("government_employment_support_procurement", 0.0))
            for row in post
        ),
        "cumulative_productivity_dividend_procurement": sum(
            float(row.get("government_productivity_dividend_procurement", 0.0))
            for row in post
        ),
        "peak_unemployment_rate": max((float(row["unemployment_rate"]) for row in post), default=0.0),
        "tail_employment_rate": mean([float(row["employment_rate"]) for row in tail]) if tail else 0.0,
        "tail_cash_vulnerable_rate": mean(
            [float(row.get("cash_vulnerable_rate", 0.0)) for row in tail]
        )
        if tail
        else 0.0,
        "tail_essential_cash_shortfall_rate": mean(
            [float(row.get("essential_cash_shortfall_rate", 0.0)) for row in tail]
        )
        if tail
        else 0.0,
        "tail_liquidity_vulnerable_rate": mean(
            [float(row.get("liquidity_vulnerable_rate", 0.0)) for row in tail]
        )
        if tail
        else 0.0,
        "tail_consumption_compression_rate": mean(
            [float(row.get("consumption_compression_rate", 0.0)) for row in tail]
        )
        if tail
        else 0.0,
        "tail_persistent_unemployment_rate": mean(
            [float(row.get("persistent_unemployment_rate", 0.0)) for row in tail]
        )
        if tail
        else 0.0,
        "tail_shock_persistent_unemployment_rate": mean(
            [
                float(row.get("shock_persistent_unemployment_rate", 0.0))
                for row in tail
            ]
        )
        if tail
        else 0.0,
        "tail_economic_stress_rate": mean(
            [float(row.get("economic_stress_rate", 0.0)) for row in tail]
        )
        if tail
        else 0.0,
        "tail_public_service_index": mean([float(row["public_service_index"]) for row in tail]) if tail else 0.0,
        "tail_market_hhi": mean(
            [float(row.get("market_hhi", 0.0)) for row in tail]
        )
        if tail
        else 0.0,
        "tail_aggressive_price_market_share": mean(
            [float(row.get("aggressive_price_market_share", 0.0)) for row in tail]
        )
        if tail
        else 0.0,
        "tail_below_cost_pricing_market_share": mean(
            [float(row.get("below_cost_pricing_market_share", 0.0)) for row in tail]
        )
        if tail
        else 0.0,
        "cumulative_personal_ai_spending": sum(
            float(row.get("personal_ai_spending", 0.0)) for row in post
        ),
        "tail_personal_ai_mean_use_rate": mean(
            [float(row.get("personal_ai_mean_use_rate", 0.0)) for row in tail]
        )
        if tail
        else 0.0,
        "cumulative_fiscal_curtailment": sum(
            float(row.get("government_fiscal_curtailment", 0.0)) for row in post
        ),
        "ending_debt_ratio": float(history[-1]["government_debt_ratio"]) if history else 0.0,
        "ending_formal_debt_ratio": float(
            history[-1].get(
                "government_formal_debt_ratio",
                history[-1]["government_debt_ratio"],
            )
        )
        if history
        else 0.0,
        "ending_government_arrears": float(
            history[-1].get("government_arrears", 0.0)
        )
        if history
        else 0.0,
        "ending_government_arrears_ratio": float(
            history[-1].get("government_arrears_ratio", 0.0)
        )
        if history
        else 0.0,
        "ending_firm_count": int(history[-1].get("firm_count", 0))
        if history
        else 0,
        "cumulative_firm_entries": int(
            history[-1].get("cumulative_firm_entries", 0)
        )
        if history
        else 0,
        "cumulative_firm_exits": int(
            history[-1].get("cumulative_firm_exits", 0)
        )
        if history
        else 0,
        "cumulative_entry_jobs": int(
            history[-1].get("cumulative_entry_jobs", 0)
        )
        if history
        else 0,
        "cumulative_exit_jobs": int(
            history[-1].get("cumulative_exit_jobs", 0)
        )
        if history
        else 0,
        "cumulative_ai_attributable_layoffs_blocked": int(
            history[-1].get(
                "cumulative_ai_attributable_layoffs_blocked", 0
            )
        )
        if history
        else 0,
        "cumulative_distress_exemption_layoffs": int(
            history[-1].get("cumulative_distress_exemption_layoffs", 0)
        )
        if history
        else 0,
        "cumulative_firm_exit_layoffs": sum(
            int(row.get("firm_exit_layoffs", 0)) for row in post
        ),
        "cumulative_retention_wage_subsidy": float(
            history[-1].get("cumulative_retention_wage_subsidy", 0.0)
        )
        if history
        else 0.0,
        "cumulative_restructuring_firm_months": sum(
            int(row.get("firms_in_restructuring", 0)) for row in post
        ),
        "peak_firms_in_restructuring": max(
            (int(row.get("firms_in_restructuring", 0)) for row in post),
            default=0,
        ),
        "tail_average_work_intensity": mean(
            [float(row.get("average_work_intensity", 1.0)) for row in tail]
        )
        if tail
        else 0.0,
        "tail_average_required_work_hours": mean(
            [
                float(row.get("average_required_work_hours", 0.0))
                for row in tail
            ]
        )
        if tail
        else 0.0,
        "cumulative_ai_levy_revenue": float(
            history[-1].get("cumulative_ai_levy_revenue", 0.0)
        )
        if history
        else 0.0,
        "cumulative_ai_levy_public_service_spending": float(
            history[-1].get(
                "cumulative_ai_levy_public_service_spending", 0.0
            )
        )
        if history
        else 0.0,
        "cumulative_ai_levy_public_investment": float(
            history[-1].get(
                "cumulative_ai_levy_public_investment", 0.0
            )
        )
        if history
        else 0.0,
        "ending_ai_levy_fund_balance": float(
            history[-1].get("government_ai_levy_fund_balance", 0.0)
        )
        if history
        else 0.0,
        "ending_ai_levy_bridge_advance": float(
            history[-1].get("government_ai_levy_bridge_advance", 0.0)
        )
        if history
        else 0.0,
        "peak_ai_levy_bridge_advance": max(
            (
                float(row.get("government_ai_levy_bridge_advance", 0.0))
                for row in post
            ),
            default=0.0,
        ),
        "tail_wage_employment_rate": mean(
            [float(row.get("wage_employment_rate", 0.0)) for row in tail]
        )
        if tail
        else 0.0,
        "tail_self_employment_rate": mean(
            [float(row.get("self_employment_rate", 0.0)) for row in tail]
        )
        if tail
        else 0.0,
        "cumulative_solo_entries": int(
            history[-1].get("cumulative_solo_entries", 0)
        )
        if history
        else 0,
        "cumulative_solo_exits": int(
            history[-1].get("cumulative_solo_exits", 0)
        )
        if history
        else 0,
        "cumulative_voluntary_wage_exits": int(
            history[-1].get("cumulative_voluntary_wage_exits", 0)
        )
        if history
        else 0,
        "cumulative_solo_enterprise_sales": sum(
            float(row.get("solo_enterprise_sales", 0.0)) for row in post
        ),
        "cumulative_solo_enterprise_income": sum(
            float(row.get("solo_enterprise_income", 0.0)) for row in post
        ),
        "cumulative_solo_substitution_sales": float(
            history[-1].get("cumulative_solo_substitution_sales", 0.0)
        )
        if history
        else 0.0,
        "cumulative_solo_b2b_sales": float(
            history[-1].get("cumulative_solo_b2b_sales", 0.0)
        )
        if history
        else 0.0,
        "cumulative_solo_induced_demand_sales": float(
            history[-1].get("cumulative_solo_induced_demand_sales", 0.0)
        )
        if history
        else 0.0,
        "cumulative_solo_external_sales": float(
            history[-1].get("cumulative_solo_external_sales", 0.0)
        )
        if history
        else 0.0,
        "cumulative_solo_net_additional_demand": sum(
            float(row.get("solo_net_additional_demand", 0.0)) for row in post
        ),
        "cumulative_solo_incumbent_displacement": sum(
            float(row.get("solo_incumbent_displacement", 0.0)) for row in post
        ),
    }
    culture_names = sorted(
        {
            key[len("culture_") : -len("_firm_count")]
            for row in history
            for key in row
            if key.startswith("culture_") and key.endswith("_firm_count")
        }
    )
    early_post = post[:12]
    for culture in culture_names:
        for metric in (
            "market_share",
            "employment_index",
            "employment_retention",
            "average_price",
            "sales",
            "retained_profit",
        ):
            source = f"culture_{culture}_{metric}"
            result[f"early_{source}"] = (
                mean([float(row.get(source, 0.0)) for row in early_post])
                if early_post
                else 0.0
            )
            result[f"tail_{source}"] = (
                mean([float(row.get(source, 0.0)) for row in tail])
                if tail
                else 0.0
            )
        result[f"cumulative_culture_{culture}_sales"] = sum(
            float(row.get(f"culture_{culture}_sales", 0.0)) for row in post
        )
        result[f"cumulative_culture_{culture}_retained_profit"] = sum(
            float(row.get(f"culture_{culture}_retained_profit", 0.0))
            for row in post
        )
    for group in INCOME_GROUPS:
        for metric in (
            "employment_rate",
            "mean_disposable_income",
            "mean_real_consumption",
            "mean_cash",
            "mean_deposits",
            "mean_managed_fund_assets",
            "mean_financial_wealth",
            "cash_vulnerable_rate",
            "essential_cash_shortfall_rate",
            "liquidity_vulnerable_rate",
            "income_stress_rate",
            "consumption_compression_rate",
            "persistent_unemployment_rate",
            "shock_persistent_unemployment_rate",
            "economic_stress_rate",
            "mean_cash_buffer_months",
            "mean_liquid_buffer_months",
        ):
            key = f"group_{group}_{metric}"
            observations = [float(row[key]) for row in tail if key in row]
            result[f"tail_{key}"] = mean(observations) if observations else 0.0
    for epsilon in ATKINSON_EPSILONS:
        label = _epsilon_label(epsilon)
        for domain in ("disposable_income", "real_consumption"):
            key = f"{domain}_atkinson_{label}"
            result[f"tail_{key}"] = mean([float(row[key]) for row in tail]) if tail else 0.0
    return result


def paired_effect(treatment: dict[str, float], control: dict[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in treatment.keys() & control.keys():
        if isinstance(treatment[key], (int, float)) and isinstance(control[key], (int, float)):
            result[key] = float(treatment[key]) - float(control[key])
    return result


def _demand_recovery_months(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    shock_month: int,
    threshold: float = 0.99,
    window_months: int = 6,
) -> int | None:
    """Return recovery timing only after demand has first fallen below control.

    A value of zero means treatment demand never fell below the threshold.  A
    value of ``None`` means it did fall below the threshold but did not later
    sustain a recovery for the required number of consecutive calendar months.
    """
    if not pairs:
        return None
    if window_months <= 0:
        raise ValueError("window_months must be positive")

    meets_threshold = [
        float(treatment["household_consumption"])
        >= threshold * float(control["household_consumption"])
        for treatment, control in pairs
    ]
    first_breach = next(
        (index for index, meets in enumerate(meets_threshold) if not meets),
        None,
    )
    if first_breach is None:
        return 0

    last_start = len(pairs) - window_months
    for index in range(first_breach + 1, last_start + 1):
        window = pairs[index : index + window_months]
        months = [int(treatment["month"]) for treatment, _ in window]
        if months != list(range(months[0], months[0] + window_months)):
            continue
        if all(meets_threshold[index : index + window_months]):
            return months[0] - shock_month
    return None


def compare_paths(
    treatment: list[dict[str, Any]], control: list[dict[str, Any]], shock_month: int = 25
) -> dict[str, Any]:
    control_by_month = {int(row["month"]): row for row in control}
    pairs = [
        (row, control_by_month[int(row["month"])])
        for row in treatment
        if int(row["month"]) >= shock_month and int(row["month"]) in control_by_month
    ]
    treatment_b60 = sum(float(t["bottom60_real_consumption"]) for t, _ in pairs)
    control_b60 = sum(float(c["bottom60_real_consumption"]) for _, c in pairs)
    recovery_month = _demand_recovery_months(pairs, shock_month)
    tail = pairs[-12:]
    peak_unemployment_delta = max(
        (float(t["unemployment_rate"]) - float(c["unemployment_rate"]) for t, c in pairs), default=0.0
    )
    result: dict[str, Any] = {
        "bottom60_cumulative_real_consumption_gain": treatment_b60 / control_b60 - 1.0 if control_b60 else None,
        "peak_unemployment_rate_delta": peak_unemployment_delta,
        "tail_employment_rate_delta": mean([float(t["employment_rate"]) - float(c["employment_rate"]) for t, c in tail]) if tail else 0.0,
        "tail_long_unemployment_rate_delta": mean([float(t["long_unemployment_rate"]) - float(c["long_unemployment_rate"]) for t, c in tail]) if tail else 0.0,
        "tail_cash_vulnerable_rate_delta": mean(
            [
                float(t.get("cash_vulnerable_rate", 0.0))
                - float(c.get("cash_vulnerable_rate", 0.0))
                for t, c in tail
            ]
        )
        if tail
        else 0.0,
        "tail_essential_cash_shortfall_rate_delta": mean(
            [
                float(t.get("essential_cash_shortfall_rate", 0.0))
                - float(c.get("essential_cash_shortfall_rate", 0.0))
                for t, c in tail
            ]
        )
        if tail
        else 0.0,
        "tail_liquidity_vulnerable_rate_delta": mean(
            [
                float(t.get("liquidity_vulnerable_rate", 0.0))
                - float(c.get("liquidity_vulnerable_rate", 0.0))
                for t, c in tail
            ]
        )
        if tail
        else 0.0,
        "tail_consumption_compression_rate_delta": mean(
            [
                float(t.get("consumption_compression_rate", 0.0))
                - float(c.get("consumption_compression_rate", 0.0))
                for t, c in tail
            ]
        )
        if tail
        else 0.0,
        "tail_economic_stress_rate_delta": mean(
            [
                float(t.get("economic_stress_rate", 0.0))
                - float(c.get("economic_stress_rate", 0.0))
                for t, c in tail
            ]
        )
        if tail
        else 0.0,
        "maximum_household_demand_gap": min((float(t["household_consumption"]) / max(float(c["household_consumption"]), 1e-9) - 1.0 for t, c in pairs), default=0.0),
        "demand_recovery_months": recovery_month,
        "tail_public_service_gain": mean([float(t["public_service_index"]) - float(c["public_service_index"]) for t, c in tail]) if tail else 0.0,
        "ending_government_debt_ratio_delta": float(pairs[-1][0]["government_debt_ratio"]) - float(pairs[-1][1]["government_debt_ratio"]) if pairs else 0.0,
        "ending_formal_debt_ratio_delta": float(
            pairs[-1][0].get(
                "government_formal_debt_ratio",
                pairs[-1][0]["government_debt_ratio"],
            )
        )
        - float(
            pairs[-1][1].get(
                "government_formal_debt_ratio",
                pairs[-1][1]["government_debt_ratio"],
            )
        )
        if pairs
        else 0.0,
        "ending_government_arrears_ratio_delta": float(
            pairs[-1][0].get("government_arrears_ratio", 0.0)
        )
        - float(pairs[-1][1].get("government_arrears_ratio", 0.0))
        if pairs
        else 0.0,
        "ending_firm_count_delta": int(
            pairs[-1][0].get("firm_count", 0)
        )
        - int(pairs[-1][1].get("firm_count", 0))
        if pairs
        else 0,
        "cumulative_firm_entries_delta": int(
            pairs[-1][0].get("cumulative_firm_entries", 0)
        )
        - int(pairs[-1][1].get("cumulative_firm_entries", 0))
        if pairs
        else 0,
        "cumulative_firm_exits_delta": int(
            pairs[-1][0].get("cumulative_firm_exits", 0)
        )
        - int(pairs[-1][1].get("cumulative_firm_exits", 0))
        if pairs
        else 0,
        "cumulative_entry_jobs_delta": int(
            pairs[-1][0].get("cumulative_entry_jobs", 0)
        )
        - int(pairs[-1][1].get("cumulative_entry_jobs", 0))
        if pairs
        else 0,
        "cumulative_exit_jobs_delta": int(
            pairs[-1][0].get("cumulative_exit_jobs", 0)
        )
        - int(pairs[-1][1].get("cumulative_exit_jobs", 0))
        if pairs
        else 0,
        "ordinary_resident_benefit_pass": bool(
            pairs
            and treatment_b60 >= control_b60
            and mean([float(t["employment_rate"]) - float(c["employment_rate"]) for t, c in tail]) >= -0.01
            and mean([float(t["long_unemployment_rate"]) - float(c["long_unemployment_rate"]) for t, c in tail]) <= 0.0
        ),
    }
    for key in (
        "cumulative_ai_attributable_layoffs_blocked",
        "cumulative_distress_exemption_layoffs",
        "cumulative_retention_wage_subsidy",
        "cumulative_ai_levy_revenue",
        "cumulative_ai_levy_public_service_spending",
        "cumulative_ai_levy_public_investment",
        "government_ai_levy_fund_balance",
        "government_ai_levy_bridge_advance",
        "cumulative_solo_entries",
        "cumulative_solo_exits",
        "cumulative_voluntary_wage_exits",
        "cumulative_solo_substitution_sales",
        "cumulative_solo_b2b_sales",
        "cumulative_solo_induced_demand_sales",
        "cumulative_solo_external_sales",
    ):
        result[f"{key}_delta"] = (
            float(pairs[-1][0].get(key, 0.0))
            - float(pairs[-1][1].get(key, 0.0))
            if pairs
            else 0.0
        )
    for key in (
        "average_work_intensity",
        "average_required_work_hours",
        "wage_employment_rate",
        "self_employment_rate",
    ):
        result[f"tail_{key}_delta"] = (
            mean(
                [
                    float(t.get(key, 0.0)) - float(c.get(key, 0.0))
                    for t, c in tail
                ]
            )
            if tail
            else 0.0
        )
    for key in (
        "solo_enterprise_sales",
        "solo_enterprise_income",
        "ai_infrastructure_levy",
    ):
        result[f"cumulative_{key}_delta"] = sum(
            float(t.get(key, 0.0)) - float(c.get(key, 0.0))
            for t, c in pairs
        )
    consumption_series = {
        "overall": "real_consumption",
        "bottom60": "bottom60_real_consumption",
        "bottom80": "bottom80_real_consumption",
        **{f"group_{group}": f"group_{group}_real_consumption" for group in INCOME_GROUPS},
    }
    for label, key in consumption_series.items():
        treatment_total = sum(float(t[key]) for t, _ in pairs)
        control_total = sum(float(c[key]) for _, c in pairs)
        output_key = "cumulative_real_consumption_gain" if label == "overall" else f"{label}_cumulative_real_consumption_gain"
        result[output_key] = treatment_total / control_total - 1.0 if control_total else None

    income_series = {
        "overall": "disposable_income",
        "bottom60": "bottom60_disposable_income",
        "bottom80": "bottom80_disposable_income",
        **{f"group_{group}": f"group_{group}_disposable_income" for group in INCOME_GROUPS},
    }
    for label, key in income_series.items():
        treatment_total = sum(float(t[key]) for t, _ in pairs)
        control_total = sum(float(c[key]) for _, c in pairs)
        output_key = "cumulative_disposable_income_gain" if label == "overall" else f"{label}_cumulative_disposable_income_gain"
        result[output_key] = treatment_total / control_total - 1.0 if control_total else None

    for group in INCOME_GROUPS:
        key = f"group_{group}_employment_rate"
        result[f"tail_{key}_delta"] = mean([float(t[key]) - float(c[key]) for t, c in tail]) if tail else 0.0
    for epsilon in ATKINSON_EPSILONS:
        epsilon_label = _epsilon_label(epsilon)
        for domain in ("disposable_income", "real_consumption"):
            key = f"{domain}_atkinson_{epsilon_label}"
            result[f"tail_{key}_delta"] = mean([float(t[key]) - float(c[key]) for t, c in tail]) if tail else 0.0

    extra_spending = sum(float(t["government_spending"]) - float(c["government_spending"]) for t, c in pairs)
    consumption_improvement = treatment_b60 - control_b60
    reduced_long_unemployment_months = sum(
        (float(c["long_unemployment_rate"]) - float(t["long_unemployment_rate"])) * float(t["population"])
        for t, c in pairs
    )
    result["cumulative_incremental_government_spending"] = extra_spending
    result["cost_per_bottom60_real_consumption_gain"] = (
        extra_spending / consumption_improvement if consumption_improvement > 0.0 else None
    )
    result["cost_per_reduced_long_unemployment_month"] = (
        extra_spending / reduced_long_unemployment_months if reduced_long_unemployment_months > 0.0 else None
    )
    return result


def _bootstrap_mean_interval(values: list[float], samples: int, confidence: float, seed: str) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    if len(values) == 1 or samples <= 0:
        return values[0], values[0]
    rng = random.Random(seed)
    size = len(values)
    bootstrapped = [mean([values[rng.randrange(size)] for _ in range(size)]) for _ in range(samples)]
    alpha = (1.0 - confidence) / 2.0
    return quantile(bootstrapped, alpha), quantile(bootstrapped, 1.0 - alpha)


def aggregate_comparisons(
    comparisons: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 2000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 20260716,
) -> dict[str, dict[str, float]]:
    if not comparisons:
        return {}
    excluded = {"population", "seed", "scenario", "control", "demand_recovery_months", "ordinary_resident_benefit_pass"}
    numeric_keys = sorted({
        key
        for row in comparisons
        for key, value in row.items()
        if key not in excluded and isinstance(value, (int, float)) and not isinstance(value, bool)
    })
    result: dict[str, dict[str, float]] = {}
    for key in numeric_keys:
        values = [float(row[key]) for row in comparisons if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)]
        if not values:
            continue
        standard_error = stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
        q1 = quantile(values, 0.25)
        q3 = quantile(values, 0.75)
        ci_low, ci_high = _bootstrap_mean_interval(
            values,
            bootstrap_samples,
            bootstrap_confidence,
            f"{bootstrap_seed}:{key}:{len(values)}",
        )
        result[key] = {
            "count": len(values),
            "mean": mean(values),
            "median": median(values),
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
            "standard_error": standard_error,
            "bootstrap_confidence": bootstrap_confidence,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "positive_seed_share": sum(value >= 0 for value in values) / len(values),
        }
    recoveries = [row["demand_recovery_months"] for row in comparisons if row.get("demand_recovery_months") is not None]
    result["demand_recovery"] = {
        "recovered_seed_share": len(recoveries) / len(comparisons),
        "mean_months_when_recovered": mean(recoveries) if recoveries else -1.0,
    }
    result["ordinary_resident_benefit"] = {
        "pass_seed_share": sum(bool(row["ordinary_resident_benefit_pass"]) for row in comparisons) / len(comparisons)
    }
    return result
