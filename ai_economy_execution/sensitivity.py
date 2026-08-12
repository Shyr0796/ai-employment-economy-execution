from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .configuration import load_config
from .experiments import parse_integer_set, run_numeric_experiment
from .reporting import write_records_csv
from .result_layout import (
    DEFAULT_MATRIX_ROOT,
    VALID_RESULT_STAGES,
    matrix_aggregate_dir,
)


SENSITIVITY_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "private_ai_productivity",
        "label": "私人AI生产率增量倍率",
        "kind": "ai_target_scale",
        "low": 0.5,
        "center": 1.0,
        "high": 1.5,
    },
    {
        "id": "household_cash_buffer",
        "label": "居民现金缓冲倍率",
        "kind": "household_cash_scale",
        "low": 0.5,
        "center": 1.0,
        "high": 2.0,
    },
    {
        "id": "firm_cash_buffer",
        "label": "企业现金缓冲月数",
        "kind": "path",
        "path": "firms.cash_months",
        "low": 1.0,
        "center": 3.0,
        "high": 6.0,
    },
    {
        "id": "precautionary_consumption",
        "label": "失业预防性消费收缩",
        "kind": "path",
        "path": "households.precautionary_response",
        "low": 0.05,
        "center": 0.10,
        "high": 0.20,
    },
    {
        "id": "labor_adjustment_speed",
        "label": "企业用工调整速度",
        "kind": "path",
        "path": "firms.labor_adjustment_speed",
        "low": 0.10,
        "center": 0.20,
        "high": 0.30,
    },
    {
        "id": "price_productivity_pass_through",
        "label": "AI生产率价格传导",
        "kind": "path",
        "path": "firms.price_productivity_pass_through",
        "low": 0.35,
        "center": 0.55,
        "high": 0.75,
    },
    {
        "id": "transfer_response",
        "label": "政府转移响应强度",
        "kind": "path",
        "path": "government.transfer_response_rate",
        "low": 0.25,
        "center": 0.50,
        "high": 0.75,
    },
    {
        "id": "procurement_response",
        "label": "政府采购响应强度",
        "kind": "path",
        "path": "government.procurement_response_rate",
        "low": 0.15,
        "center": 0.30,
        "high": 0.45,
    },
    {
        "id": "bankruptcy_cash_distress_months",
        "label": "企业现金资不抵债退出月数",
        "kind": "path",
        "path": "firms.bankruptcy_cash_distress_months",
        "low": 3.0,
        "center": 6.0,
        "high": 9.0,
    },
    {
        "id": "entry_unemployment_threshold",
        "label": "新企业进入所需最低失业率",
        "kind": "path",
        "path": "firms.entry_unemployment_threshold",
        "low": 0.005,
        "center": 0.010,
        "high": 0.020,
    },
)

SENSITIVITY_METRICS = (
    "bottom60_cumulative_real_consumption_gain",
    "bottom80_cumulative_real_consumption_gain",
    "peak_unemployment_rate_delta",
    "tail_employment_rate_delta",
    "tail_cash_vulnerable_rate_delta",
    "tail_essential_cash_shortfall_rate_delta",
    "tail_liquidity_vulnerable_rate_delta",
    "tail_consumption_compression_rate_delta",
    "tail_economic_stress_rate_delta",
    "maximum_household_demand_gap",
    "tail_public_service_gain",
    "ending_government_debt_ratio_delta",
    "ending_formal_debt_ratio_delta",
    "ending_government_arrears_ratio_delta",
    "ending_firm_count_delta",
    "cumulative_firm_entries_delta",
    "cumulative_firm_exits_delta",
    "cumulative_entry_jobs_delta",
    "cumulative_exit_jobs_delta",
    "tail_disposable_income_atkinson_1_0_delta",
    "tail_real_consumption_atkinson_1_0_delta",
)


def _set_path(config: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    target = config
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def apply_sensitivity_value(config: dict[str, Any], spec: dict[str, Any], value: float) -> dict[str, Any]:
    modified = copy.deepcopy(config)
    if spec["kind"] == "path":
        _set_path(modified, spec["path"], value)
    elif spec["kind"] == "ai_target_scale":
        for firm_type in modified["firms"]["types"]:
            firm_type["ai_target"] = 1.0 + (float(firm_type["ai_target"]) - 1.0) * value
    elif spec["kind"] == "household_cash_scale":
        for group in modified["households"]["income_groups"]:
            group["cash_months"] = float(group["cash_months"]) * value
    else:
        raise ValueError(f"Unknown sensitivity kind: {spec['kind']}")
    return modified


def _extract_effect_rows(
    variant: dict[str, Any],
    baseline: dict[str, Any],
    spec: dict[str, Any],
    level: str,
    value: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for population, scenarios in variant["aggregate"].items():
        for scenario, metrics in scenarios.items():
            baseline_metrics = baseline["aggregate"][population][scenario]
            for metric in SENSITIVITY_METRICS:
                statistics = metrics.get(metric)
                baseline_statistics = baseline_metrics.get(metric)
                if not statistics or not baseline_statistics:
                    continue
                variant_mean = float(statistics["mean"])
                baseline_mean = float(baseline_statistics["mean"])
                rows.append({
                    "parameter": spec["id"],
                    "parameter_label": spec["label"],
                    "level": level,
                    "value": value,
                    "population": int(population),
                    "scenario": scenario,
                    "metric": metric,
                    "baseline_mean": baseline_mean,
                    "variant_mean": variant_mean,
                    "change_from_baseline": variant_mean - baseline_mean,
                    "variant_median": statistics.get("median"),
                    "variant_q1": statistics.get("q1"),
                    "variant_q3": statistics.get("q3"),
                    "variant_bootstrap_ci_low": statistics.get("bootstrap_ci_low"),
                    "variant_bootstrap_ci_high": statistics.get("bootstrap_ci_high"),
                    "variant_positive_seed_share": statistics.get("positive_seed_share"),
                })
    return rows


def _write_sensitivity_report(rows: list[dict[str, Any]], output: Path, design: dict[str, Any]) -> None:
    primary = [row for row in rows if row["metric"] == "bottom60_cumulative_real_consumption_gain"]
    lines = [
        "# 正式单因素敏感性分析",
        "",
        f"- 参数数量：{len(design['parameters'])}",
        f"- 边界变体：{len(design['parameters']) * 2}个（中心值复用主实验）",
        f"- 人口规模：{', '.join(str(value) for value in design['populations'])}",
        f"- 随机种子：{len(design['seeds'])}组",
        "- 每个变体均运行E0—E6，并在种子内进行配对比较。",
        "",
        "## 底部60%累计实际消费敏感性",
        "",
        "|参数|边界|取值|人口|情景|主实验均值|变体均值|变化量|Bootstrap区间|",
        "|---|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in primary:
        lines.append(
            f"|{row['parameter_label']}|{row['level']}|{row['value']}|{row['population']}|{row['scenario']}|"
            f"{row['baseline_mean']:.6f}|{row['variant_mean']:.6f}|{row['change_from_baseline']:.6f}|"
            f"[{float(row['variant_bootstrap_ci_low']):.6f}, {float(row['variant_bootstrap_ci_high']):.6f}]|"
        )
    lines.extend([
        "",
        "完整指标包括就业、需求缺口、公共服务、债务和收入/消费Atkinson指数，见 `sensitivity_effects.csv`。",
        "",
    ])
    (output / "sensitivity_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_sensitivity_analysis(
    config_path: Path | None,
    populations: list[int],
    seeds: list[int],
    output: Path,
    *,
    baseline_config: dict[str, Any] | None = None,
    baseline_result: dict[str, Any] | None = None,
    bootstrap_samples: int = 2000,
    bootstrap_confidence: float = 0.95,
    scenario_definition_version: str | None = None,
) -> dict[str, Any]:
    base = copy.deepcopy(baseline_config) if baseline_config is not None else load_config(config_path)
    output.mkdir(parents=True, exist_ok=True)
    if baseline_result is None:
        baseline_result = run_numeric_experiment(
            None,
            populations,
            seeds,
            output / "baseline",
            baseline_config=base,
            write_paths=False,
            bootstrap_samples=bootstrap_samples,
            bootstrap_confidence=bootstrap_confidence,
            generate_artifacts=False,
            scenario_definition_version=scenario_definition_version,
        )

    config_dir = output / "variant_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for spec in SENSITIVITY_SPECS:
        for level in ("low", "high"):
            value = float(spec[level])
            variant_config = apply_sensitivity_value(base, spec, value)
            (config_dir / f"{spec['id']}_{level}.json").write_text(
                json.dumps(variant_config, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            variant_result = run_numeric_experiment(
                None,
                populations,
                seeds,
                output / "variants" / f"{spec['id']}_{level}",
                baseline_config=variant_config,
                write_paths=False,
                bootstrap_samples=bootstrap_samples,
                bootstrap_confidence=bootstrap_confidence,
                generate_artifacts=False,
                scenario_definition_version=scenario_definition_version,
            )
            rows.extend(_extract_effect_rows(variant_result, baseline_result, spec, level, value))

    design = {
        "method": "one_at_a_time",
        "center_reuses_main_experiment": True,
        "populations": populations,
        "seeds": seeds,
        "scenarios": list(base["scenarios"]),
        "scenario_definition_version": (
            scenario_definition_version
            or base.get("default_scenario_definition_version", "unversioned")
        ),
        "metrics": list(SENSITIVITY_METRICS),
        "parameters": list(SENSITIVITY_SPECS),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_confidence": bootstrap_confidence,
    }
    result = {"design": design, "effects": rows}
    (output / "sensitivity_design.json").write_text(json.dumps(design, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "sensitivity_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_records_csv(rows, output / "sensitivity_effects.csv")
    _write_sensitivity_report(rows, output, design)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run formal one-at-a-time sensitivity analysis")
    parser.add_argument("--populations", default="500,1000,5000")
    parser.add_argument("--seeds", default="1-50")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--scenario-definition-version", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument(
        "--result-stage",
        choices=VALID_RESULT_STAGES,
        default="formal",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    args = parser.parse_args()
    config = load_config(args.config)
    scenario_definition_version = (
        args.scenario_definition_version
        or config.get("default_scenario_definition_version", "unversioned")
    )
    output = args.output or matrix_aggregate_dir(
        root=args.matrix_root,
        stage=args.result_stage,
        scenario_definition_version=scenario_definition_version,
        cognitive_regime="R0",
        provider="offline",
        model=None,
        analysis="sensitivity",
        populations=args.populations,
        seeds=args.seeds,
    )
    result = run_sensitivity_analysis(
        args.config,
        parse_integer_set(args.populations),
        parse_integer_set(args.seeds),
        output,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_confidence=args.bootstrap_confidence,
        scenario_definition_version=args.scenario_definition_version,
    )
    print(json.dumps({"effects": len(result["effects"]), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
