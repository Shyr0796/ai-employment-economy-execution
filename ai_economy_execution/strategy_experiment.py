from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .configuration import load_config, scenario_config
from .core import EconomyEngine
from .experiments import parse_integer_set
from .initialization import initialize_economy
from .metrics import summarize, validate_metric, write_history


CULTURES = ("augmentation", "cost_cutter", "adaptive")
FIRM_REGIMES = CULTURES + (
    "mixed_no_competition",
    "mixed_price_only",
    "mixed_capacity_reputation",
    "mixed_competition",
)
STRATEGIES = (
    "passive_safety_net",
    "active_demand",
    "productivity_dividend",
    "fiscal_guard",
    "active_demand_regulation",
)


def _apply_firm_regime(config: dict[str, Any], regime: str) -> None:
    firms = config["firms"]
    competition = firms["competition"]
    if regime in CULTURES:
        firms["culture_mode"] = regime
        competition["enabled"] = True
        return
    firms["culture_mode"] = "mixed"
    competition["enabled"] = regime != "mixed_no_competition"
    if regime == "mixed_price_only":
        competition["capacity_weight"] = 1.0
        competition["employment_reputation_weight"] = 0.0
    elif regime == "mixed_capacity_reputation":
        competition["price_sensitivity"] = 0.0


def _run_path(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state = initialize_economy(config)
    engine = EconomyEngine(state, config)
    for _ in range(int(config["simulation"]["months"])):
        validate_metric(engine.step())
    return state.history, summarize(
        state.history, int(config["simulation"]["shock_month"])
    )


def _effect(summary: dict[str, Any], control: dict[str, Any]) -> dict[str, float]:
    def pct(key: str) -> float:
        denominator = max(abs(float(control[key])), 1e-9)
        return 100.0 * (float(summary[key]) - float(control[key])) / denominator

    return {
        "real_consumption_gain_pct": pct("cumulative_real_consumption"),
        "bottom60_real_consumption_gain_pct": pct(
            "bottom60_cumulative_real_consumption"
        ),
        "tail_employment_gain_pp": 100.0
        * (
            float(summary["tail_employment_rate"])
            - float(control["tail_employment_rate"])
        ),
        "peak_unemployment_change_pp": 100.0
        * (
            float(summary["peak_unemployment_rate"])
            - float(control["peak_unemployment_rate"])
        ),
        "real_government_purchase_gain_pct": pct(
            "cumulative_real_government_purchase"
        ),
        "ending_liability_ratio_change_pp": 100.0
        * (
            float(summary["ending_debt_ratio"])
            - float(control["ending_debt_ratio"])
        ),
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    numeric_keys = (
        "real_consumption_gain_pct",
        "bottom60_real_consumption_gain_pct",
        "tail_employment_gain_pp",
        "peak_unemployment_change_pp",
        "real_government_purchase_gain_pct",
        "cumulative_real_government_procurement",
        "cumulative_employment_support_procurement",
        "cumulative_productivity_dividend_procurement",
        "ending_liability_ratio_change_pp",
        "ending_debt_ratio",
        "ending_government_arrears_ratio",
        "tail_market_hhi",
        "tail_aggressive_price_market_share",
        "tail_below_cost_pricing_market_share",
        "cumulative_personal_ai_spending",
        "tail_personal_ai_mean_use_rate",
        "cumulative_fiscal_curtailment",
        "cumulative_firm_exits",
    )
    numeric_keys += tuple(
        f"{period}_culture_{culture}_{metric}"
        for culture in CULTURES
        for period in ("early", "tail")
        for metric in (
            "market_share",
            "employment_index",
            "employment_retention",
            "average_price",
            "sales",
            "retained_profit",
        )
    )
    numeric_keys += tuple(
        f"cumulative_culture_{culture}_{metric}"
        for culture in CULTURES
        for metric in ("sales", "retained_profit")
    )
    for culture in FIRM_REGIMES:
        for strategy in STRATEGIES:
            selected = [
                row
                for row in rows
                if row["culture"] == culture and row["strategy"] == strategy
            ]
            aggregate: dict[str, Any] = {
                "culture": culture,
                "strategy": strategy,
                "n_seeds": len(selected),
            }
            for key in numeric_keys:
                values = [float(row[key]) for row in selected]
                aggregate[f"{key}_mean"] = mean(values) if values else 0.0
                aggregate[f"{key}_sd"] = pstdev(values) if len(values) > 1 else 0.0
            result.append(aggregate)
    return result


DIAGNOSTIC_KEYS = (
    "real_consumption_gain_pct",
    "bottom60_real_consumption_gain_pct",
    "tail_employment_gain_pp",
    "peak_unemployment_change_pp",
    "ending_debt_ratio",
    "tail_market_hhi",
    "tail_aggressive_price_market_share",
    "tail_below_cost_pricing_market_share",
    "cumulative_firm_exits",
)


def _aggregate_paired(
    rows: list[dict[str, Any]], group_keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    aggregates: list[dict[str, Any]] = []
    for group, selected in groups.items():
        aggregate = dict(zip(group_keys, group))
        aggregate["n_seeds"] = len(selected)
        for key in DIAGNOSTIC_KEYS:
            values = [float(row[key]) for row in selected]
            aggregate[f"{key}_mean"] = mean(values)
            aggregate[f"{key}_sd"] = pstdev(values) if len(values) > 1 else 0.0
            aggregate[f"{key}_positive_seeds"] = sum(value > 0 for value in values)
            aggregate[f"{key}_negative_seeds"] = sum(value < 0 for value in values)
        aggregates.append(aggregate)
    return aggregates


def _competition_diagnostics(
    runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = {
        (int(row["seed"]), str(row["culture"]), str(row["strategy"])): row
        for row in runs
    }
    paired: list[dict[str, Any]] = []
    channels = {
        "price_only": "mixed_price_only",
        "capacity_reputation": "mixed_capacity_reputation",
        "full_competition": "mixed_competition",
    }
    seeds = sorted({int(row["seed"]) for row in runs})
    for seed in seeds:
        for strategy in STRATEGIES:
            control = by_key[(seed, "mixed_no_competition", strategy)]
            channel_rows: dict[str, dict[str, Any]] = {}
            for channel, regime in channels.items():
                treatment = by_key[(seed, regime, strategy)]
                row: dict[str, Any] = {
                    "seed": seed,
                    "strategy": strategy,
                    "channel": channel,
                }
                for key in DIAGNOSTIC_KEYS:
                    row[key] = float(treatment[key]) - float(control[key])
                paired.append(row)
                channel_rows[channel] = row
            interaction = {
                "seed": seed,
                "strategy": strategy,
                "channel": "price_x_capacity_reputation_interaction",
            }
            for key in DIAGNOSTIC_KEYS:
                interaction[key] = (
                    channel_rows["full_competition"][key]
                    - channel_rows["price_only"][key]
                    - channel_rows["capacity_reputation"][key]
                )
            paired.append(interaction)
    return paired, _aggregate_paired(paired, ("strategy", "channel"))


def _policy_diagnostics(
    runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = {
        (int(row["seed"]), str(row["culture"]), str(row["strategy"])): row
        for row in runs
    }
    paired: list[dict[str, Any]] = []
    seeds = sorted({int(row["seed"]) for row in runs})
    for seed in seeds:
        for regime in FIRM_REGIMES:
            control = by_key[(seed, regime, "passive_safety_net")]
            for strategy in STRATEGIES:
                treatment = by_key[(seed, regime, strategy)]
                row: dict[str, Any] = {
                    "seed": seed,
                    "culture": regime,
                    "strategy": strategy,
                    "control": "passive_safety_net",
                }
                for key in DIAGNOSTIC_KEYS:
                    row[key] = float(treatment[key]) - float(control[key])
                paired.append(row)
    return paired, _aggregate_paired(paired, ("culture", "strategy"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_analysis(
    output: Path,
    aggregate: list[dict[str, Any]],
    validation: dict[str, Any],
) -> None:
    lines = [
        "# 企业文化 × 政府需求管理：探索性结果",
        "",
        "本实验不含出口部门，只解释模型内的国内就业—收入—消费—企业需求—财政反馈。结果不是现实因果估计。",
        "",
        "## 使用边界",
        "",
        "- 每个处理组合都使用同种子、同企业制度的 E0 匹配对照；不能跨制度直接把原始水平当作政策效果。",
        "- `competition_paired.csv` 分解价格通道、产能/就业声誉通道及二者交互；`policy_paired.csv` 比较政府策略。",
        "- 企业文化是行为参数组合，不是道德标签；个人 AI 是居民自费的工作投入，已从福利消费中剔除。",
        "- 主动需求管理、生产率红利采购和反恶性竞争监管为分离策略，避免把机制混在一个综合包中。",
        "- 尚未加入出口、行业投入产出与现实参数校准，不能外推为国家预测。",
        "",
        "## 自动校验",
        "",
        f"- 最大销售恒等式残差：{validation['max_abs_sales_identity_error']:.3e}",
        f"- 最大工资恒等式残差：{validation['max_abs_wage_identity_error']:.3e}",
        f"- 最大税收恒等式残差：{validation['max_abs_tax_identity_error']:.3e}",
        f"- 最大银行资产负债表残差：{validation['max_abs_bank_balance_sheet_error']:.3e}",
        f"- 冲击前匹配对照最大差异：{validation['max_abs_pretrend_gap']:.3e}",
        f"- 支持采购超过总采购的月数：{validation['support_exceeds_procurement_months']}",
        "",
    ]
    output.joinpath("analysis.md").write_text("\n".join(lines), encoding="utf-8")


def _source_hashes(config_path: Path | None) -> dict[str, str]:
    root = Path(__file__).resolve().parent
    files = [
        root / "core.py",
        root / "models.py",
        root / "initialization.py",
        root / "configuration.py",
        root / "metrics.py",
        root / "strategy_experiment.py",
        config_path.resolve() if config_path else root / "config" / "baseline.json",
    ]
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }


def _validation_summary(
    histories: list[list[dict[str, Any]]],
    treatment_controls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    shock_month: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "sales_identity_error",
        "wage_identity_error",
        "tax_identity_error",
        "bank_balance_sheet_error",
    ):
        result[f"max_abs_{key}"] = max(
            (abs(float(row.get(key, 0.0))) for history in histories for row in history),
            default=0.0,
        )
    pretrend_fields = (
        "employment_rate",
        "real_consumption",
        "aggregate_price",
        "firm_sales",
    )
    result["max_abs_pretrend_gap"] = max(
        (
            abs(float(treatment.get(key, 0.0)) - float(control.get(key, 0.0)))
            for treatment_history, control_history in treatment_controls
            for treatment, control in zip(treatment_history, control_history)
            if int(treatment["month"]) < shock_month
            for key in pretrend_fields
        ),
        default=0.0,
    )
    result["support_exceeds_procurement_months"] = sum(
        float(row.get("government_employment_support_procurement", 0.0))
        + float(row.get("government_productivity_dividend_procurement", 0.0))
        > float(row.get("government_procurement", 0.0)) + 1e-8
        for history in histories
        for row in history
    )
    return result


def run_strategy_experiment(
    *,
    population: int,
    seeds: list[int],
    months: int,
    output: Path,
    config_path: Path | None = None,
    write_paths: bool = True,
    scenario_definition_version: str | None = None,
) -> dict[str, Any]:
    baseline = load_config(config_path)
    output.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now().astimezone().isoformat()
    rows: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    control_paths: dict[tuple[int, str], tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    histories: list[list[dict[str, Any]]] = []
    treatment_controls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    for seed in seeds:
        for regime in FIRM_REGIMES:
            control_config = scenario_config(
                baseline,
                "E0",
                population,
                seed,
                scenario_definition_version=scenario_definition_version,
            )
            control_config["simulation"]["months"] = months
            _apply_firm_regime(control_config, regime)
            control_config["government"]["policy_strategy"] = "passive_safety_net"
            control_history, control_summary = _run_path(control_config)
            control_paths[(seed, regime)] = (control_history, control_summary)
            histories.append(control_history)
            controls.append({
                "population": population,
                "seed": seed,
                "culture": regime,
                "scenario": "E0",
                **control_summary,
            })
            if write_paths:
                write_history(
                    control_history,
                    output / "paths" / f"seed_{seed}" / regime / "control_E0",
                )

        for culture in FIRM_REGIMES:
            control_history, control_summary = control_paths[(seed, culture)]
            for strategy in STRATEGIES:
                config = scenario_config(
                    baseline,
                    "E5",
                    population,
                    seed,
                    scenario_definition_version=scenario_definition_version,
                )
                config["simulation"]["months"] = months
                _apply_firm_regime(config, culture)
                config["government"]["policy_strategy"] = strategy
                history, summary = _run_path(config)
                histories.append(history)
                treatment_controls.append((history, control_history))
                row = {
                    "population": population,
                    "seed": seed,
                    "culture": culture,
                    "strategy": strategy,
                    **summary,
                    **_effect(summary, control_summary),
                }
                rows.append(row)
                if write_paths:
                    write_history(
                        history,
                        output / "paths" / f"seed_{seed}" / culture / strategy,
                    )

    aggregate = _aggregate(rows)
    competition_paired, competition_diagnostics = _competition_diagnostics(rows)
    policy_paired, policy_diagnostics = _policy_diagnostics(rows)
    shock_month = int(baseline["simulation"]["shock_month"])
    validation = _validation_summary(
        histories, treatment_controls, shock_month
    )
    source_hashes = _source_hashes(config_path)
    result = {
        "design": {
            "population": population,
            "seeds": seeds,
            "months": months,
            "shock_month": shock_month,
            "cultures": list(CULTURES),
            "firm_regimes": list(FIRM_REGIMES),
            "government_strategies": list(STRATEGIES),
            "treatment_scenario": "E5",
            "scenario_definition_version": (
                scenario_definition_version
                or baseline.get(
                    "default_scenario_definition_version", "unversioned"
                )
            ),
            "control_scenario": "E0",
            "common_random_numbers": True,
            "matched_control_by_seed_and_firm_regime": True,
            "monthly_paths_saved": write_paths,
            "export_sector_included": False,
            "interpretation": "exploratory_model_comparison",
        },
        "controls": controls,
        "runs": rows,
        "aggregate": aggregate,
        "competition_paired": competition_paired,
        "competition_diagnostics": competition_diagnostics,
        "policy_paired": policy_paired,
        "policy_diagnostics": policy_diagnostics,
        "validation": validation,
        "source_hashes": source_hashes,
    }
    output.joinpath("strategy_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(output / "strategy_runs.csv", rows)
    _write_csv(output / "control_runs.csv", controls)
    _write_csv(output / "strategy_matrix.csv", aggregate)
    _write_csv(output / "competition_paired.csv", competition_paired)
    _write_csv(output / "competition_diagnostics.csv", competition_diagnostics)
    _write_csv(output / "policy_paired.csv", policy_paired)
    _write_csv(output / "policy_diagnostics.csv", policy_diagnostics)
    output.joinpath("resolved_baseline_config.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    completed_at = datetime.now().astimezone().isoformat()
    manifest = {
        "status": "complete",
        "started_at": started_at,
        "completed_at": completed_at,
        "expected_control_paths": len(seeds) * len(FIRM_REGIMES),
        "completed_control_paths": len(controls),
        "expected_treatment_paths": len(seeds) * len(FIRM_REGIMES) * len(STRATEGIES),
        "completed_treatment_paths": len(rows),
        "monthly_paths_saved": write_paths,
        "scenario_definition_version": result["design"][
            "scenario_definition_version"
        ],
        "validation": validation,
        "source_hashes": source_hashes,
    }
    output.joinpath("run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_analysis(output, aggregate, validation)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Firm culture and competition channels by five demand-management strategies"
    )
    parser.add_argument("--population", type=int, default=500)
    parser.add_argument("--seeds", default="1-5")
    parser.add_argument("--months", type=int, default=120)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--scenario-definition-version", default=None)
    parser.add_argument(
        "--no-paths",
        action="store_true",
        help="Do not persist monthly paths (paths are saved by default)",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or Path(
        "ai_economy_execution/results/research_matrix/auxiliary/"
        "strategy/culture_policy/"
        f"N{args.population:05d}_M{args.months:03d}_S{args.seeds}_{timestamp}"
    )
    result = run_strategy_experiment(
        population=args.population,
        seeds=parse_integer_set(args.seeds),
        months=args.months,
        output=output,
        config_path=args.config,
        write_paths=not args.no_paths,
        scenario_definition_version=args.scenario_definition_version,
    )
    print(
        json.dumps(
            {"output": str(output), "runs": len(result["runs"])},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
