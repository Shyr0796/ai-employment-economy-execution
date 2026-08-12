from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .configuration import load_config
from .experiments import parse_integer_set
from .result_layout import (
    COGNITIVE_REGIME_ROLES,
    DEFAULT_MATRIX_ROOT,
    VALID_RESULT_STAGES,
    matrix_cell_dir,
)


ALL_REGIMES = tuple(COGNITIVE_REGIME_ROLES)
ALL_SCENARIOS = tuple(f"E{index}" for index in range(7))
PACKAGE_ROOT = Path(__file__).resolve().parent


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "results" in path.parts or "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(PACKAGE_ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _parse_ordered_set(
    value: str, *, prefix: str, minimum: int, maximum: int
) -> list[str]:
    selected: set[int] = set()
    for item in value.split(","):
        item = item.strip().upper()
        if not item:
            continue
        if "-" in item:
            start_raw, end_raw = item.split("-", 1)
            start = int(start_raw.removeprefix(prefix))
            end = int(end_raw.removeprefix(prefix))
            selected.update(range(start, end + 1))
        else:
            selected.add(int(item.removeprefix(prefix)))
    if not selected or min(selected) < minimum or max(selected) > maximum:
        raise ValueError(
            f"{prefix} selection must be within {prefix}{minimum}-{prefix}{maximum}"
        )
    return [f"{prefix}{index}" for index in sorted(selected)]


def parse_regimes(value: str) -> list[str]:
    return _parse_ordered_set(value, prefix="R", minimum=0, maximum=3)


def parse_scenarios(value: str) -> list[str]:
    return _parse_ordered_set(value, prefix="E", minimum=0, maximum=6)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _run_status(
    output: Path,
    *,
    source_fingerprint: str,
    scenario: str,
    population: int,
    seed: int,
    requested_final_month: int,
    cognitive_regime: str,
) -> str:
    if not output.exists():
        return "missing"
    if not any(output.iterdir()):
        return "missing"
    manifest_path = output / "run_manifest.json"
    resolved_path = output / "resolved_config.json"
    if not manifest_path.is_file() or not resolved_path.is_file():
        return "incompatible"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "incompatible"
    design = manifest.get("design", {})
    execution = resolved.get("execution", {})
    complete = bool(
        manifest.get("status") == "completed"
        and manifest.get("completion", {}).get("complete")
        and manifest.get("provenance", {}).get("source_fingerprint")
        == source_fingerprint
        and design.get("scenario") == scenario
        and int(design.get("population", -1)) == population
        and int(design.get("seed", -1)) == seed
        and int(design.get("requested_final_month", -1))
        == requested_final_month
        and execution.get("cognitive_regime") == cognitive_regime
    )
    return "completed" if complete else "incompatible"


def _append_optional_runtime_args(
    command: list[str], args: argparse.Namespace, *, paid: bool
) -> None:
    command.extend(
        [
            "--batch-size",
            str(args.batch_size),
            "--max-fallback-rate",
            str(args.max_fallback_rate),
            "--max-role-fallback-rate",
            str(args.max_role_fallback_rate),
            "--result-stage",
            args.result_stage,
        ]
    )
    if args.config is not None:
        command.extend(["--config", str(args.config)])
    if args.llm_max_workers is not None:
        command.extend(["--llm-max-workers", str(args.llm_max_workers)])
    if args.llm_concurrency is not None:
        command.extend(["--llm-concurrency", str(args.llm_concurrency)])
    if args.replay:
        command.append("--replay")
    if paid and args.require_response_model_match:
        command.append("--require-response-model-match")


def _equilibrium_command(
    args: argparse.Namespace, *, seed: int, output: Path, warmup_months: int
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "ai_economy_execution.run",
        "--scenario",
        "E0",
        "--scenario-definition-version",
        args.scenario_definition_version,
        "--population",
        str(args.population),
        "--months",
        str(warmup_months),
        "--seed",
        str(seed),
        "--provider",
        "offline",
        "--llm-roles",
        "",
        "--cognitive-regime",
        "R0",
        "--output",
        str(output),
    ]
    _append_optional_runtime_args(command, args, paid=False)
    return command


def _branch_command(
    args: argparse.Namespace,
    *,
    regime: str,
    scenario: str,
    seed: int,
    output: Path,
    checkpoint: Path,
) -> list[str]:
    roles = ",".join(COGNITIVE_REGIME_ROLES[regime])
    paid = regime != "R0"
    command = [
        sys.executable,
        "-m",
        "ai_economy_execution.run",
        "--scenario",
        scenario,
        "--scenario-definition-version",
        args.scenario_definition_version,
        "--population",
        str(args.population),
        "--months",
        str(args.months),
        "--seed",
        str(seed),
        "--provider",
        args.provider if paid else "offline",
        "--llm-roles",
        roles,
        "--cognitive-regime",
        regime,
        "--initial-state",
        str(checkpoint),
        "--output",
        str(output),
    ]
    if paid:
        command.extend(["--activate-cognitive-regime-from-checkpoint"])
        if args.model is not None:
            command.extend(["--model", args.model])
        if args.key_env is not None:
            command.extend(["--key-env", args.key_env])
        if args.api_base is not None:
            command.extend(["--api-base", args.api_base])
    _append_optional_runtime_args(command, args, paid=paid)
    return command


def _preflight_command(args: argparse.Namespace, regimes: list[str]) -> list[str] | None:
    paid_regimes = [regime for regime in regimes if regime != "R0"]
    if not paid_regimes:
        return None
    roles = sorted(
        {
            role
            for regime in paid_regimes
            for role in COGNITIVE_REGIME_ROLES[regime]
        }
    )
    command = [
        sys.executable,
        "-m",
        "ai_economy_execution.api_preflight",
        "--provider",
        args.provider,
        "--llm-roles",
        ",".join(roles),
    ]
    if args.model is not None:
        command.extend(["--model", args.model])
    if args.key_env is not None:
        command.extend(["--key-env", args.key_env])
    if args.api_base is not None:
        command.extend(["--api-base", args.api_base])
    return command


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    warmup_months = int(config["simulation"]["warmup_months"])
    shock_month = int(config["simulation"]["shock_month"])
    if args.months <= warmup_months:
        raise ValueError(
            f"--months must exceed the {warmup_months}-month equilibrium"
        )
    if shock_month != warmup_months + 1:
        raise ValueError(
            "Formal cognitive activation requires shock_month == warmup_months + 1"
        )
    regimes = parse_regimes(args.regimes)
    scenarios = parse_scenarios(args.scenarios)
    seeds = parse_integer_set(args.seeds)
    source_fingerprint = _source_fingerprint()
    tasks: list[dict[str, Any]] = []
    for seed in seeds:
        r0_cell = matrix_cell_dir(
            root=args.matrix_root,
            stage=args.result_stage,
            scenario_definition_version=args.scenario_definition_version,
            cognitive_regime="R0",
            provider="offline",
            model=None,
            population=args.population,
            months=args.months,
            seed=seed,
        )
        equilibrium = r0_cell / "equilibrium"
        tasks.append(
            {
                "id": f"S{seed:03d}:equilibrium",
                "kind": "equilibrium",
                "regime": "R0",
                "scenario": "E0",
                "seed": seed,
                "paid": False,
                "output": str(equilibrium),
                "status": _run_status(
                    equilibrium,
                    source_fingerprint=source_fingerprint,
                    scenario="E0",
                    population=args.population,
                    seed=seed,
                    requested_final_month=warmup_months,
                    cognitive_regime="R0",
                ),
                "command": _equilibrium_command(
                    args,
                    seed=seed,
                    output=equilibrium,
                    warmup_months=warmup_months,
                ),
            }
        )
        checkpoint = equilibrium / "pre_equilibrium_state.json"
        for regime in regimes:
            paid = regime != "R0"
            provider = args.provider if paid else "offline"
            model = args.model if paid else None
            cell = matrix_cell_dir(
                root=args.matrix_root,
                stage=args.result_stage,
                scenario_definition_version=args.scenario_definition_version,
                cognitive_regime=regime,
                provider=provider,
                model=model,
                population=args.population,
                months=args.months,
                seed=seed,
            )
            for scenario in scenarios:
                output = cell / scenario
                tasks.append(
                    {
                        "id": f"S{seed:03d}:{regime}:{scenario}",
                        "kind": "cell",
                        "regime": regime,
                        "scenario": scenario,
                        "seed": seed,
                        "paid": paid,
                        "output": str(output),
                        "status": _run_status(
                            output,
                            source_fingerprint=source_fingerprint,
                            scenario=scenario,
                            population=args.population,
                            seed=seed,
                            requested_final_month=args.months,
                            cognitive_regime=regime,
                        ),
                        "command": _branch_command(
                            args,
                            regime=regime,
                            scenario=scenario,
                            seed=seed,
                            output=output,
                            checkpoint=checkpoint,
                        ),
                    }
                )
    orchestration = (
        args.matrix_root
        / args.result_stage
        / args.scenario_definition_version
        / "orchestration"
        / (
            f"N{args.population:05d}_M{args.months:03d}"
            f"_S{min(seeds):03d}-{max(seeds):03d}"
        )
        / (
            f"{'-'.join(regimes)}__{'-'.join(scenarios)}"
        )
    )
    return {
        "schema_version": 1,
        "status": "planned",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_fingerprint": source_fingerprint,
        "protocol": {
            "name": "common_R0_equilibrium_then_month25_cognitive_activation",
            "warmup_months": warmup_months,
            "activation_month": shock_month,
            "scenario_definition_version": args.scenario_definition_version,
        },
        "design": {
            "population": args.population,
            "months": args.months,
            "seeds": seeds,
            "regimes": regimes,
            "scenarios": scenarios,
            "provider": args.provider,
            "model": args.model,
            "result_stage": args.result_stage,
        },
        "counts": {
            "equilibria": len(seeds),
            "cells": len(seeds) * len(regimes) * len(scenarios),
            "paid_cells": len(seeds)
            * len([regime for regime in regimes if regime != "R0"])
            * len(scenarios),
        },
        "orchestration_dir": str(orchestration),
        "preflight_command": _preflight_command(args, regimes),
        "tasks": tasks,
    }


def execute_plan(plan: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    orchestration = Path(plan["orchestration_dir"])
    manifest_path = orchestration / "matrix_manifest.json"
    plan["status"] = "running"
    plan["started_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(manifest_path, plan)

    preflight = plan.get("preflight_command")
    if preflight is not None:
        completed = subprocess.run(
            preflight,
            check=False,
            text=True,
            capture_output=True,
        )
        preflight_record = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        _atomic_write_json(orchestration / "api_preflight.json", preflight_record)
        if completed.returncode != 0:
            plan["status"] = "preflight_failed"
            plan["error"] = completed.stderr.strip() or completed.stdout.strip()
            plan["finished_at"] = datetime.now(timezone.utc).isoformat()
            _atomic_write_json(manifest_path, plan)
            raise RuntimeError("API preflight failed; inspect api_preflight.json")

    for task in plan["tasks"]:
        if task["status"] == "completed":
            if args.resume:
                task["execution_status"] = "skipped_completed"
                _atomic_write_json(manifest_path, plan)
                continue
            raise FileExistsError(
                f"Completed output already exists and --no-resume was used: {task['output']}"
            )
        if task["status"] == "incompatible":
            plan["status"] = "blocked_existing_output"
            plan["error"] = (
                "Existing non-empty output is incomplete or belongs to a "
                f"different source/design: {task['output']}"
            )
            _atomic_write_json(manifest_path, plan)
            raise FileExistsError(plan["error"])
        task["execution_status"] = "running"
        task["started_at"] = datetime.now(timezone.utc).isoformat()
        log_path = orchestration / "logs" / (
            task["id"].replace(":", "__") + ".log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        task["log"] = str(log_path)
        _atomic_write_json(manifest_path, plan)
        print(
            f"[RUN] {task['id']} -> {task['output']} "
            f"(log: {log_path})",
            flush=True,
        )
        with log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                task["command"],
                check=False,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        task["returncode"] = completed.returncode
        task["finished_at"] = datetime.now(timezone.utc).isoformat()
        if completed.returncode != 0:
            task["execution_status"] = "failed"
            plan["status"] = "failed"
            plan["error"] = f"Task failed: {task['id']}"
            plan["finished_at"] = datetime.now(timezone.utc).isoformat()
            _atomic_write_json(manifest_path, plan)
            raise RuntimeError(
                f"{plan['error']}; inspect {log_path}"
            )
        task["execution_status"] = "completed"
        print(f"[DONE] {task['id']}", flush=True)
        _atomic_write_json(manifest_path, plan)

    plan["status"] = "completed"
    plan["finished_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(manifest_path, plan)
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or run the R0-R3 x E0-E6 matrix from one shared offline "
            "R0 month-24 equilibrium per seed"
        )
    )
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--months", type=int, default=36)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--regimes", default="R0-R3")
    parser.add_argument("--scenarios", default="E0-E6")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--scenario-definition-version",
        default="institutional_v2",
    )
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument(
        "--result-stage",
        choices=VALID_RESULT_STAGES,
        default="smoke",
    )
    parser.add_argument(
        "--provider",
        choices=["hkust", "deepseek", "dashscope", "moonshot", "openai", "custom"],
        default="hkust",
    )
    parser.add_argument("--model", default="gpt-3.5-turbo")
    parser.add_argument("--key-env", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--llm-max-workers", type=int, default=2)
    parser.add_argument("--llm-concurrency", type=int, default=2)
    parser.add_argument("--max-fallback-rate", type=float, default=0.01)
    parser.add_argument("--max-role-fallback-rate", type=float, default=0.01)
    parser.add_argument(
        "--require-response-model-match",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--replay", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip completed matching cells and stop on incompatible partial output",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute tasks; without this flag only a read-only plan is printed",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plan = build_plan(args)
    if not args.execute:
        printable = {
            key: value
            for key, value in plan.items()
            if key not in {"tasks", "preflight_command"}
        }
        printable["preflight_command"] = plan["preflight_command"]
        printable["tasks"] = [
            {
                "id": task["id"],
                "paid": task["paid"],
                "status": task["status"],
                "output": task["output"],
            }
            for task in plan["tasks"]
        ]
        print(json.dumps(printable, ensure_ascii=False, indent=2))
        return
    result = execute_plan(plan, args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "counts": result["counts"],
                "orchestration_dir": result["orchestration_dir"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
