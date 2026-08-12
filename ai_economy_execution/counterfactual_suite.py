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


SCENARIOS = ("E0", "E1", "E5")


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
    (output / "counterfactual_comparisons.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fieldnames = sorted({key for row in rows for key in row})
    with (output / "counterfactual_comparisons.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


async def run_counterfactual_suite(args: argparse.Namespace) -> dict[str, Any]:
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
        raise FileExistsError(f"Counterfactual suite output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    warmup_months = int(config["simulation"]["warmup_months"])
    if args.months <= warmup_months:
        raise ValueError(f"--months must be greater than the warmup length {warmup_months}")

    manifest: dict[str, Any] = {
        "status": "running",
        "design": "one LLM-active pre-equilibrium followed by E0/E1/E5 common-state forks",
        "population": args.population,
        "seed": args.seed,
        "months": args.months,
        "provider": args.provider,
        "scenario_definition_version": scenario_definition_version,
        "cognitive_regime": cognitive_regime,
        "result_layout_version": RESULT_LAYOUT_VERSION,
        "result_stage": getattr(args, "result_stage", "smoke"),
        "matrix_cell": str(output),
        "llm_roles": sorted(item.strip() for item in args.llm_roles.split(",") if item.strip()),
        "scenarios": list(SCENARIOS),
        "runs": {},
    }
    manifest_path = output / "suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
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
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        shock_month = int(load_config(args.config)["simulation"]["shock_month"])
        comparisons = [
            {
                "treatment": "E1",
                "control": "E0",
                "estimand": "AI shock effect",
                **compare_paths(histories["E1"], histories["E0"], shock_month),
            },
            {
                "treatment": "E5",
                "control": "E1",
                "estimand": "combined policy incremental effect",
                **compare_paths(histories["E5"], histories["E1"], shock_month),
            },
            {
                "treatment": "E5",
                "control": "E0",
                "estimand": "AI plus combined policy total effect",
                **compare_paths(histories["E5"], histories["E0"], shock_month),
            },
        ]
        _write_comparisons(comparisons, output)
        manifest["comparisons"] = comparisons
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run E0/E1/E5 from one qualified LLM-active month-24 checkpoint"
    )
    parser.add_argument("--population", type=int, default=500)
    parser.add_argument("--months", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--scenario-definition-version",
        default=None,
        help=(
            "Select E5 semantics: institutional_v2 (default) or legacy_v1"
        ),
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
        choices=["offline", "hkust", "deepseek", "dashscope", "moonshot", "openai", "custom"],
        default="hkust",
    )
    parser.add_argument(
        "--llm-roles",
        default="resident,firm,government",
        help="Comma-separated: resident,firm,government",
    )
    parser.add_argument("--key-env", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--model", default=None)
    return parser


def main() -> None:
    result = asyncio.run(run_counterfactual_suite(build_parser().parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
