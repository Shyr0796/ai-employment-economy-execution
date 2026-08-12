from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "baseline.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG
    return json.loads(config_path.read_text(encoding="utf-8"))


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def scenario_config(
    config: dict[str, Any],
    scenario: str,
    population: int | None = None,
    seed: int | None = None,
    scenario_definition_version: str | None = None,
) -> dict[str, Any]:
    versions = config.get("scenario_definition_versions", {})
    default_version = config.get("default_scenario_definition_version")
    selected_version = scenario_definition_version or default_version
    scenarios = copy.deepcopy(config["scenarios"])
    version_metadata: dict[str, Any] | None = None
    if versions:
        if not selected_version:
            raise ValueError(
                "default_scenario_definition_version is required when "
                "scenario_definition_versions are configured"
            )
        if selected_version not in versions:
            raise ValueError(
                f"Unknown scenario definition version {selected_version!r}; "
                f"expected one of {sorted(versions)}"
            )
        version_metadata = copy.deepcopy(versions[selected_version])
        scenarios = deep_merge(
            scenarios,
            version_metadata.get("scenarios", {}),
        )
    elif scenario_definition_version is not None:
        raise ValueError(
            "The selected config does not define scenario_definition_versions"
        )
    if scenario not in scenarios:
        raise ValueError(
            f"Unknown scenario {scenario!r}; expected one of {sorted(scenarios)}"
        )
    result = copy.deepcopy(config)
    result["scenarios"] = scenarios
    result["scenario_definition_version"] = selected_version or "unversioned"
    if version_metadata is not None:
        result["scenario_definition_description"] = str(
            version_metadata.get("description", "")
        )
    result["active_scenario"] = scenario
    result["scenario"] = copy.deepcopy(scenarios[scenario])
    institutional = result.get("institutional_experiments", {})
    if result["scenario"].get("employment_responsibility", False):
        responsibility = institutional.get("employment_responsibility", {})
        share = float(responsibility.get("protected_shadow_employment_share", 1.0))
        if not 0.0 <= share <= 1.0:
            raise ValueError(
                "protected_shadow_employment_share must be between 0 and 1"
            )
        cost_share = float(responsibility.get("wage_cost_sharing_rate", 0.0))
        if not 0.0 <= cost_share <= 1.0:
            raise ValueError("wage_cost_sharing_rate must be between 0 and 1")
        for key in (
            "wage_cost_sharing_months",
            "distress_extension_months",
            "early_warning_loss_months",
            "early_warning_cash_months",
            "restructuring_grace_months",
        ):
            if int(responsibility.get(key, 0)) < 0:
                raise ValueError(f"employment-responsibility {key} must be nonnegative")
    if result["scenario"].get("ai_infrastructure_levy", False):
        levy = institutional.get("ai_infrastructure_levy", {})
        initial_capture_rate = float(
            levy.get("initial_capture_rate", levy.get("capture_rate", 0.0))
        )
        capture_rate = float(levy.get("capture_rate", 0.0))
        pass_through = float(levy.get("consumer_price_pass_through", 1.0))
        exemption_share = float(
            levy.get("basic_consumption_exemption_share", 0.0)
        )
        advance_share = float(levy.get("same_month_advance_share", 0.0))
        allocation = float(levy.get("public_service_share", 0.0)) + float(
            levy.get("public_investment_share", 0.0)
        )
        if not 0.0 <= initial_capture_rate <= capture_rate <= 1.0:
            raise ValueError(
                "AI levy capture rates must satisfy 0 <= initial <= target <= 1"
            )
        if not 0.0 <= pass_through <= 1.0:
            raise ValueError(
                "AI levy consumer_price_pass_through must be between 0 and 1"
            )
        if not 0.0 <= exemption_share <= 1.0:
            raise ValueError(
                "AI levy basic_consumption_exemption_share must be between 0 and 1"
            )
        if not 0.0 <= advance_share <= 1.0:
            raise ValueError(
                "AI levy same_month_advance_share must be between 0 and 1"
            )
        if not 0.0 <= allocation <= 1.0 + 1e-12:
            raise ValueError("AI levy earmark shares must sum to at most 1")
    if result["scenario"].get("solo_enterprise", False):
        solo = institutional.get("solo_enterprise", {})
        for key in (
            "max_monthly_entry_share",
            "max_self_employed_share",
            "solo_market_share",
            "substitution_market_share",
            "b2b_investment_share",
            "induced_demand_baseline_share",
            "induced_cash_drawdown_rate",
            "external_demand_baseline_share",
            "operating_cost_rate",
            "income_tax_rate",
        ):
            value = float(solo.get(key, 0.0))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"solo-enterprise {key} must be between 0 and 1")
    policy_strategy = result["scenario"].get("policy_strategy")
    if result["scenario"].get("procurement_response", False) and policy_strategy is None:
        raise ValueError(
            f"Scenario {scenario!r} enables procurement_response but does not "
            "assign a post-shock policy_strategy"
        )
    if policy_strategy is not None:
        strategies = result["government"].get("policy_strategies", {})
        if policy_strategy not in strategies:
            raise ValueError(
                f"Scenario {scenario!r} references unknown government policy strategy "
                f"{policy_strategy!r}"
            )
        selected = strategies[policy_strategy]
        if result["scenario"].get("procurement_response", False) and not any(
            float(selected.get(key, 0.0)) > 0.0
            for key in (
                "procurement_response_multiplier",
                "employment_support_rate",
                "productivity_dividend_rate",
            )
        ):
            raise ValueError(
                f"Scenario {scenario!r} enables procurement_response but policy "
                f"strategy {policy_strategy!r} has no active procurement channel"
            )
        # EconomyEngine._government_strategy() forces the common
        # passive_safety_net before shock_month, so this is a post-shock
        # treatment assignment rather than a pre-trend change.
        result["government"]["policy_strategy"] = policy_strategy
    if population is not None:
        if population < 10:
            raise ValueError("population must be at least 10")
        result["simulation"]["population"] = population
    if seed is not None:
        result["simulation"]["seed"] = seed
    pre_shock_government: dict[str, Any] = {}
    for key in ("max_annual_deficit_ratio", "debt_limit_ratio"):
        if key in result["scenario"]:
            pre_shock_government[key] = copy.deepcopy(result["government"][key])
            result["government"][key] = result["scenario"][key]
    if pre_shock_government:
        # Scenario-specific fiscal constraints are treatment variables.  Keep
        # their common pre-shock values so the core can activate them only at
        # shock_month while still exposing the resolved post-shock config.
        result["pre_shock_government"] = pre_shock_government
    return result
