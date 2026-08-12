from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path
from typing import Any

from .configuration import load_config
from .metrics import compare_paths
from .result_layout import (
    DEFAULT_MATRIX_ROOT,
    RESULT_LAYOUT_VERSION,
    VALID_RESULT_STAGES,
    matrix_cell_dir,
    resolve_cognitive_regime,
)
from .run import run_agentsociety


SCENARIOS = ("E0", "E1", "E2", "E3", "E4")
PRETREATMENT_METADATA_KEYS = {"scenario"}


def _run_args(args: argparse.Namespace, **overrides: Any) -> argparse.Namespace:
    values = {
        "scenario": "E0",
        "population": args.population,
        "months": args.months,
        "seed": args.seed,
        "config": args.config,
        "scenario_definition_version": getattr(
            args, "scenario_definition_version", None
        ),
        "output": args.output,
        "batch_size": args.batch_size,
        "llm_max_workers": args.llm_max_workers,
        "llm_concurrency": args.llm_concurrency,
        "max_fallback_rate": args.max_fallback_rate,
        "max_role_fallback_rate": args.max_role_fallback_rate,
        "require_response_model_match": args.require_response_model_match,
        "replay": args.replay,
        "provider": args.provider,
        "llm_roles": args.llm_roles,
        "key_env": args.key_env,
        "api_base": args.api_base,
        "model": args.model,
        "initial_state": None,
        "allow_unstable_equilibrium": False,
        "allow_checkpoint_source_mismatch": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _write_comparisons(rows: list[dict[str, Any]], output: Path) -> None:
    (output / "institutional_comparisons.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fieldnames = sorted({key for row in rows for key in row})
    with (output / "institutional_comparisons.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _validate_suite(
    histories: dict[str, list[dict[str, Any]]], shock_month: int
) -> dict[str, Any]:
    control_pre = [
        row for row in histories["E0"] if int(row["month"]) < shock_month
    ]
    prehistory_mismatches: list[dict[str, Any]] = []
    comparable_cells = 0
    for scenario in SCENARIOS[1:]:
        candidate_pre = [
            row
            for row in histories[scenario]
            if int(row["month"]) < shock_month
        ]
        for control, candidate in zip(control_pre, candidate_pre):
            keys = sorted(
                (set(control) & set(candidate)) - PRETREATMENT_METADATA_KEYS
            )
            for key in keys:
                comparable_cells += 1
                if candidate[key] != control[key]:
                    prehistory_mismatches.append(
                        {
                            "scenario": scenario,
                            "month": int(control["month"]),
                            "key": key,
                            "control": control[key],
                            "candidate": candidate[key],
                        }
                    )
    accounting_maxima = {}
    for scenario, history in histories.items():
        accounting_maxima[scenario] = max(
            (
                abs(float(row.get(key, 0.0)))
                for row in history
                for key in (
                    "sales_identity_error",
                    "wage_identity_error",
                    "tax_identity_error",
                    "bank_balance_sheet_error",
                )
            ),
            default=0.0,
        )

    e2_post = [
        row for row in histories["E2"] if int(row["month"]) >= shock_month
    ]
    e3_post = [
        row for row in histories["E3"] if int(row["month"]) >= shock_month
    ]
    e4_post = [
        row for row in histories["E4"] if int(row["month"]) >= shock_month
    ]
    e3_fund_error = max(
        (
            abs(
                float(row.get("cumulative_ai_levy_revenue", 0.0))
                - float(
                    row.get(
                        "cumulative_ai_levy_public_service_spending", 0.0
                    )
                )
                - float(
                    row.get(
                        "cumulative_ai_levy_public_investment", 0.0
                    )
                )
                - float(row.get("government_ai_levy_fund_balance", 0.0))
            )
            for row in e3_post
        ),
        default=0.0,
    )
    e4_labor_status_error = max(
        (
            abs(
                int(row.get("wage_employment", 0))
                + int(row.get("self_employment", 0))
                + round(
                    float(row.get("unemployment_rate", 0.0))
                    * int(row.get("population", 0))
                )
                - int(row.get("population", 0))
            )
            for row in e4_post
        ),
        default=0,
    )
    e4_channel_identity_error = max(
        (
            abs(
                float(row.get("solo_enterprise_sales", 0.0))
                - float(row.get("solo_substitution_sales", 0.0))
                - float(row.get("solo_b2b_sales", 0.0))
                - float(row.get("solo_induced_demand_sales", 0.0))
                - float(row.get("solo_external_sales", 0.0))
            )
            for row in e4_post
        ),
        default=0.0,
    )
    checks = {
        "common_pretreatment_history": not prehistory_mismatches,
        "accounting": max(accounting_maxima.values(), default=0.0) <= 1e-5,
        "e2_mechanism_activated": bool(
            e2_post
            and max(
                int(
                    row.get(
                        "cumulative_ai_attributable_layoffs_blocked", 0
                    )
                )
                for row in e2_post
            )
            > 0
            and max(
                float(row.get("cumulative_retention_wage_subsidy", 0.0))
                for row in e2_post
            )
            > 0.0
            and min(
                float(row.get("average_required_work_hours", 160.0))
                for row in e2_post
            )
            < 160.0
        ),
        "e3_levy_and_fund_identity": bool(
            e3_post
            and max(
                float(row.get("cumulative_ai_levy_revenue", 0.0))
                for row in e3_post
            )
            > 0.0
            and e3_fund_error <= 1e-6
        ),
        "e4_solo_enterprise_and_labor_status": bool(
            e4_post
            and max(int(row.get("cumulative_solo_entries", 0)) for row in e4_post)
            > 0
            and e4_labor_status_error == 0
            and e4_channel_identity_error <= 1e-6
            and max(
                float(row.get("cumulative_solo_induced_demand_sales", 0.0))
                + float(row.get("cumulative_solo_external_sales", 0.0))
                for row in e4_post
            )
            > 0.0
        ),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "common_pretreatment_comparable_cells": comparable_cells,
        "common_pretreatment_mismatch_count": len(prehistory_mismatches),
        "common_pretreatment_mismatches": prehistory_mismatches[:20],
        "maximum_accounting_error_by_scenario": accounting_maxima,
        "e3_maximum_fund_identity_error": e3_fund_error,
        "e4_maximum_labor_status_error": e4_labor_status_error,
        "e4_maximum_channel_identity_error": e4_channel_identity_error,
    }


async def run_institutional_suite(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    scenario_definition_version = (
        getattr(args, "scenario_definition_version", None)
        or config.get("default_scenario_definition_version", "unversioned")
    )
    cognitive_regime = resolve_cognitive_regime(
        args.llm_roles, getattr(args, "cognitive_regime", None)
    )
    output_arg = getattr(args, "output", None)
    if output_arg is None:
        output_arg = matrix_cell_dir(
            root=getattr(args, "matrix_root", DEFAULT_MATRIX_ROOT),
            stage=getattr(args, "result_stage", "smoke"),
            scenario_definition_version=scenario_definition_version,
            cognitive_regime=cognitive_regime,
            provider=args.provider,
            model=args.model,
            population=args.population,
            months=args.months,
            seed=args.seed,
        )
    output = Path(output_arg).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Institutional suite output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    warmup_months = int(config["simulation"]["warmup_months"])
    if args.months <= warmup_months:
        raise ValueError(
            f"--months must be greater than the warmup length {warmup_months}"
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "design": (
            "one qualified pre-equilibrium followed by common-state E0/E1/"
            "E2-retention/E3-levy/E4-solo-enterprise forks"
        ),
        "population": args.population,
        "seed": args.seed,
        "months": args.months,
        "provider": args.provider,
        "scenario_definition_version": scenario_definition_version,
        "cognitive_regime": cognitive_regime,
        "result_layout_version": RESULT_LAYOUT_VERSION,
        "result_stage": getattr(args, "result_stage", "smoke"),
        "matrix_cell": str(output),
        "llm_roles": sorted(
            item.strip() for item in args.llm_roles.split(",") if item.strip()
        ),
        "scenarios": list(SCENARIOS),
        "runs": {},
    }
    manifest_path = output / "suite_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        equilibrium_dir = output / "equilibrium"
        equilibrium = await run_agentsociety(
            _run_args(
                args,
                scenario="E0",
                months=warmup_months,
                output=equilibrium_dir,
            )
        )
        manifest["equilibrium"] = equilibrium
        checkpoint = equilibrium_dir / "pre_equilibrium_state.json"

        histories: dict[str, list[dict[str, Any]]] = {}
        for scenario in SCENARIOS:
            branch_dir = output / scenario
            result = await run_agentsociety(
                _run_args(
                    args,
                    scenario=scenario,
                    output=branch_dir,
                    initial_state=checkpoint,
                )
            )
            manifest["runs"][scenario] = result
            histories[scenario] = json.loads(
                (branch_dir / "metrics.json").read_text(encoding="utf-8")
            )
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        shock_month = int(config["simulation"]["shock_month"])
        comparisons = [
            {
                "treatment": "E1",
                "control": "E0",
                "estimand": "laissez-faire private-AI effect",
                **compare_paths(histories["E1"], histories["E0"], shock_month),
            }
        ]
        labels = {
            "E2": "employment responsibility plus wage-cost sharing effect",
            "E3": "graduated AI-rent levy plus immediate recycling effect",
            "E4": "mixed-demand solo-enterprise incremental effect",
        }
        comparisons.extend(
            {
                "treatment": scenario,
                "control": "E1",
                "estimand": labels[scenario],
                **compare_paths(
                    histories[scenario], histories["E1"], shock_month
                ),
            }
            for scenario in ("E2", "E3", "E4")
        )
        _write_comparisons(comparisons, output)
        validation = _validate_suite(histories, shock_month)
        (output / "smoke_validation.json").write_text(
            json.dumps(validation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest["comparisons"] = comparisons
        manifest["validation"] = validation
        manifest["status"] = (
            "completed" if validation["pass"] else "validation_failed"
        )
        if not validation["pass"]:
            raise RuntimeError(
                "Institutional suite validation failed: "
                + ", ".join(
                    key for key, passed in validation["checks"].items() if not passed
                )
            )
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the E0-E4 institutional experiment from one qualified "
            "month-24 checkpoint"
        )
    )
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--months", type=int, default=36)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--scenario-definition-version",
        default=None,
        help="Select versioned E5/E6 semantics recorded in the suite manifest",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Expert override. By default the path is generated under "
            "results/research_matrix using the matrix_v1 naming convention."
        ),
    )
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=DEFAULT_MATRIX_ROOT,
        help="Root directory for automatically named R x E matrix results",
    )
    parser.add_argument(
        "--result-stage",
        choices=VALID_RESULT_STAGES,
        default="smoke",
    )
    parser.add_argument(
        "--cognitive-regime",
        choices=["R0", "R1", "R2", "R3"],
        default=None,
        help="Optional assertion; it must match --llm-roles",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--llm-max-workers", type=int, default=None)
    parser.add_argument("--llm-concurrency", type=int, default=None)
    parser.add_argument("--max-fallback-rate", type=float, default=0.01)
    parser.add_argument("--max-role-fallback-rate", type=float, default=0.01)
    parser.add_argument("--require-response-model-match", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument(
        "--provider",
        choices=[
            "offline",
            "hkust",
            "deepseek",
            "dashscope",
            "moonshot",
            "openai",
            "custom",
        ],
        default="offline",
    )
    parser.add_argument(
        "--llm-roles",
        default="",
        help="Comma-separated: resident,firm,government",
    )
    parser.add_argument("--key-env", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--model", default=None)
    return parser


def main() -> None:
    result = asyncio.run(run_institutional_suite(build_parser().parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
