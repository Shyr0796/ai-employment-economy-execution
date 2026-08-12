from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .configuration import load_config, scenario_config
from .core import EconomyEngine
from .gates import (
    audit_counterfactual_paths,
    audit_history,
    gate_thresholds,
    initial_state_fingerprint,
    summarize_run_gates,
)
from .initialization import initialize_economy
from .metrics import aggregate_comparisons, compare_paths, summarize, validate_metric, write_history
from .reporting import write_experiment_artifacts
from .result_layout import (
    DEFAULT_MATRIX_ROOT,
    VALID_RESULT_STAGES,
    matrix_aggregate_dir,
)


def parse_integer_set(value: str) -> list[int]:
    values: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = (int(part) for part in item.split("-", 1))
            if end < start:
                raise ValueError(f"Invalid descending range: {item}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(item))
    unique = list(dict.fromkeys(values))
    if not unique:
        raise ValueError("At least one integer is required")
    return unique


def run_numeric_experiment(
    config_path: Path | None,
    populations: list[int],
    seeds: list[int],
    output: Path,
    *,
    baseline_config: dict[str, Any] | None = None,
    write_paths: bool = True,
    bootstrap_samples: int = 2000,
    bootstrap_confidence: float = 0.95,
    generate_artifacts: bool = True,
    scenario_definition_version: str | None = None,
) -> dict[str, Any]:
    baseline = copy.deepcopy(baseline_config) if baseline_config is not None else load_config(config_path)
    selected_scenario_version = (
        scenario_definition_version
        or baseline.get("default_scenario_definition_version", "unversioned")
    )
    scenario_names = list(baseline["scenarios"])
    records = []
    comparisons = []
    interactions = []
    path_audits = []
    counterfactual_audits = []
    thresholds = gate_thresholds(baseline)
    warmup_months = int(baseline["simulation"]["warmup_months"])
    for population in populations:
        for seed in seeds:
            histories: dict[str, list[dict[str, Any]]] = {}
            fingerprints: dict[str, str] = {}
            for scenario in scenario_names:
                config = scenario_config(
                    baseline,
                    scenario,
                    population,
                    seed,
                    scenario_definition_version=scenario_definition_version,
                )
                state = initialize_economy(config)
                fingerprints[scenario] = initial_state_fingerprint(state)
                engine = EconomyEngine(state, config)
                for _ in range(int(config["simulation"]["months"])):
                    validate_metric(engine.step())
                run_dir = output / f"N{population}" / f"seed_{seed}" / scenario
                if write_paths:
                    write_history(state.history, run_dir)
                summary = summarize(state.history, int(config["simulation"]["shock_month"]))
                histories[scenario] = state.history
                records.append({"population": population, "seed": seed, "scenario": scenario, **summary})
                path_audits.append(audit_history(
                    state.history,
                    population=population,
                    seed=seed,
                    scenario=scenario,
                    warmup_months=warmup_months,
                    thresholds=thresholds,
                ))

            counterfactual_audits.extend(audit_counterfactual_paths(
                histories,
                fingerprints,
                population=population,
                seed=seed,
                warmup_months=warmup_months,
                thresholds=thresholds,
            ))

            seed_comparisons = []
            for scenario in scenario_names:
                control_scenario = "E0" if scenario in {"E0", "E1"} else "E1"
                comparison = {
                    "population": population,
                    "seed": seed,
                    "scenario": scenario,
                    "control": control_scenario,
                    **compare_paths(
                        histories[scenario],
                        histories[control_scenario],
                        int(baseline["simulation"]["shock_month"]),
                    ),
                }
                comparisons.append(comparison)
                seed_comparisons.append(comparison)
            primary_values = {
                row["scenario"]: row["bottom60_cumulative_real_consumption_gain"]
                for row in seed_comparisons
            }
            if all(scenario in primary_values for scenario in ("E2", "E3", "E4", "E5")):
                interactions.append({
                    "population": population,
                    "seed": seed,
                    "bottom60_policy_interaction": (
                        primary_values["E5"] - primary_values["E2"] - primary_values["E3"] - primary_values["E4"]
                    ),
                })

    aggregate = {}
    for population in populations:
        aggregate[str(population)] = {}
        for scenario in scenario_names:
            selected = [row for row in comparisons if row["population"] == population and row["scenario"] == scenario]
            aggregate[str(population)][scenario] = aggregate_comparisons(
                selected,
                bootstrap_samples=bootstrap_samples,
                bootstrap_confidence=bootstrap_confidence,
            )
    result = {
        "design": {
            "populations": populations,
            "seeds": seeds,
            "scenarios": scenario_names,
            "scenario_definition_version": selected_scenario_version,
            "months": int(baseline["simulation"]["months"]),
            "shock_month": int(baseline["simulation"]["shock_month"]),
            "common_random_numbers": True,
        },
        "bootstrap": {"samples": bootstrap_samples, "confidence": bootstrap_confidence, "paired_by_seed": True},
        "runs": records,
        "comparisons": comparisons,
        "aggregate": aggregate,
        "policy_interactions": interactions,
        "run_gates": summarize_run_gates(path_audits, counterfactual_audits, thresholds),
    }
    if generate_artifacts:
        write_experiment_artifacts(result, output, baseline)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast common-random-number experiment over the shared numeric core")
    parser.add_argument("--populations", default="500,1000,5000")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--scenario-definition-version", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument(
        "--result-stage",
        choices=VALID_RESULT_STAGES,
        default="pilot",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--no-paths", action="store_true", help="Do not retain per-run monthly CSV/JSON paths")
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
        analysis="numeric_experiment",
        populations=args.populations,
        seeds=args.seeds,
    )
    result = run_numeric_experiment(
        args.config,
        parse_integer_set(args.populations),
        parse_integer_set(args.seeds),
        output,
        write_paths=not args.no_paths,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_confidence=args.bootstrap_confidence,
        scenario_definition_version=args.scenario_definition_version,
    )
    print(json.dumps({"runs": len(result["runs"]), "comparisons": len(result["comparisons"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
