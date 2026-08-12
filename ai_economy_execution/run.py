from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .behavior_audit import write_behavior_audit
from .configuration import load_config, scenario_config
from .initialization import build_agent_specs, initialize_economy
from .gates import audit_history, gate_thresholds
from .metrics import summarize, write_history
from .models import EconomyState
from .providers import (
    configure_agentsociety,
    response_model_match_kind,
    response_model_matching_policy,
)
from .result_layout import (
    DEFAULT_MATRIX_ROOT,
    RESULT_LAYOUT_VERSION,
    VALID_RESULT_STAGES,
    matrix_cell_dir,
    resolve_cognitive_regime,
)


PACKAGE_ROOT = Path(__file__).resolve().parent


def _positive_int(value: object, name: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be at least 1")
    return parsed


def _configure_llm_runtime(args: argparse.Namespace) -> dict[str, int]:
    """Resolve concurrency before AgentSociety imports its environment config."""
    if args.llm_max_workers is not None:
        os.environ["AGENTSOCIETY_LLM_RAY_MAX_WORKERS"] = str(
            _positive_int(args.llm_max_workers, "--llm-max-workers")
        )
    if args.llm_concurrency is not None:
        os.environ["AGENTSOCIETY_LLM_RAY_CONCURRENCY"] = str(
            _positive_int(args.llm_concurrency, "--llm-concurrency")
        )
    workers = _positive_int(
        os.getenv("AGENTSOCIETY_LLM_RAY_MAX_WORKERS", os.cpu_count() or 1),
        "AGENTSOCIETY_LLM_RAY_MAX_WORKERS",
    )
    concurrency = _positive_int(
        os.getenv("AGENTSOCIETY_LLM_RAY_CONCURRENCY", 16),
        "AGENTSOCIETY_LLM_RAY_CONCURRENCY",
    )
    return {
        "max_workers": workers,
        "initial_concurrency_per_worker": concurrency,
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _finalize_step_metadata(
    output_dir: Path,
    *,
    status: str,
    requested_steps: int,
    completed_steps: int,
    final_model_month: int,
) -> dict:
    """Replace ambiguous framework counters with an authoritative terminal record."""
    path = output_dir / "SOCIETY_STEP.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        payload = {}
    payload.update({
        "step_count": completed_steps,
        "completed_step_count": completed_steps,
        "requested_step_count": requested_steps,
        "final_model_month": final_model_month,
        "status": status,
        "terminated": True,
        "authoritative_writer": "ai_economy_execution.run",
        "finalized_at": datetime.now(timezone.utc).isoformat(),
    })
    _atomic_write_json(path, payload)
    return payload


def _write_run_manifest(
    output_dir: Path,
    *,
    status: str,
    started_at: str,
    source_fingerprint: str,
    resolved_config_path: Path,
    scenario: str,
    scenario_definition_version: str = "unversioned",
    population: int,
    seed: int,
    starting_month: int,
    requested_final_month: int,
    completed_history_months: int,
    completed_steps: int,
    decision_audit: dict,
    error: str | None = None,
) -> dict:
    resolved_config_hash = hashlib.sha256(resolved_config_path.read_bytes()).hexdigest()
    try:
        agentsociety_version = importlib.metadata.version("agentsociety2")
    except importlib.metadata.PackageNotFoundError:
        agentsociety_version = "unknown"
    manifest = {
        "schema_version": 1,
        "status": status,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "design": {
            "scenario": scenario,
            "scenario_definition_version": scenario_definition_version,
            "population": population,
            "seed": seed,
            "starting_month": starting_month,
            "requested_final_month": requested_final_month,
        },
        "completion": {
            "completed_history_months": completed_history_months,
            "completed_steps_this_run": completed_steps,
            "simulation_complete": completed_history_months == requested_final_month,
            "complete": status == "completed" and completed_history_months == requested_final_month,
        },
        "provenance": {
            "source_fingerprint": source_fingerprint,
            "resolved_config_sha256": resolved_config_hash,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "agentsociety2": agentsociety_version,
        },
        "decision_audit": decision_audit,
        "error": error,
    }
    _atomic_write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "results" in path.parts or "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(PACKAGE_ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _pre_equilibrium_config_fingerprint(config: dict) -> str:
    """Hash only settings that are allowed to affect the common pre-shock path."""
    payload = copy.deepcopy(config)
    for key, value in payload.get("pre_shock_government", {}).items():
        payload["government"][key] = value
    # EconomyEngine always uses passive_safety_net before shock_month even
    # when a scenario assigns a different post-shock policy strategy.
    payload.get("government", {})["policy_strategy"] = "passive_safety_net"
    payload.pop("active_scenario", None)
    payload.pop("scenario", None)
    payload.pop("pre_shock_government", None)
    payload.get("simulation", {}).pop("months", None)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _restore_pre_equilibrium_checkpoint(
    checkpoint_path: Path,
    *,
    config: dict,
    scenario: str,
    llm_roles: set[str],
    provider_info: dict[str, str],
    source_fingerprint: str,
    allow_unstable: bool,
    allow_source_mismatch: bool,
    allow_cognitive_activation: bool = False,
) -> tuple[EconomyState, dict, dict]:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Pre-equilibrium checkpoint not found: {checkpoint_path}")
    checkpoint_dir = checkpoint_path.parent
    audit_path = checkpoint_dir / "pre_equilibrium_audit.json"
    resolved_path = checkpoint_dir / "resolved_config.json"
    if not audit_path.is_file() or not resolved_path.is_file():
        raise ValueError(
            "Checkpoint must be accompanied by pre_equilibrium_audit.json and resolved_config.json"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not bool(audit.get("path_gate_pass")) and not allow_unstable:
        raise ValueError("Checkpoint did not pass the pre-equilibrium path gate")
    origin = json.loads(resolved_path.read_text(encoding="utf-8"))
    execution = origin.get("execution", {})
    origin_source = execution.get("source_fingerprint")
    if origin_source != source_fingerprint and not allow_source_mismatch:
        raise ValueError(
            "Checkpoint source fingerprint differs from the current code; "
            "use --allow-checkpoint-source-mismatch only for diagnostics"
        )
    origin_pre_config = execution.get("pre_equilibrium_config_fingerprint")
    current_pre_config = _pre_equilibrium_config_fingerprint(config)
    if origin_pre_config != current_pre_config:
        raise ValueError("Checkpoint pre-equilibrium configuration differs from the requested run")
    origin_roles = set(execution.get("llm_roles", []))
    origin_provider = execution.get("provider", {})
    cognitive_activation = None
    if origin_roles != llm_roles:
        if not allow_cognitive_activation:
            raise ValueError("Checkpoint LLM roles differ from the requested run")
        if (
            origin_roles
            or origin_provider.get("provider") != "offline"
            or not llm_roles
            or provider_info.get("provider") == "offline"
        ):
            raise ValueError(
                "Cognitive activation requires an R0 offline checkpoint and "
                "a non-empty R1/R2/R3 target role set"
            )
        warmup_months = int(config["simulation"]["warmup_months"])
        shock_month = int(config["simulation"]["shock_month"])
        if shock_month != warmup_months + 1:
            raise ValueError(
                "Cognitive activation must occur on the first post-warmup "
                f"month; warmup={warmup_months}, shock={shock_month}"
            )
        cognitive_activation = {
            "from_regime": "R0",
            "from_llm_roles": [],
            "from_provider": origin_provider,
            "to_llm_roles": sorted(llm_roles),
            "to_provider": provider_info,
            "activation_month": shock_month,
        }
    elif origin_provider != provider_info:
        raise ValueError("Checkpoint provider/model settings differ from the requested run")

    state = EconomyState.from_dict(json.loads(checkpoint_path.read_text(encoding="utf-8")))
    warmup_months = int(config["simulation"]["warmup_months"])
    if state.month != warmup_months or len(state.history) != warmup_months:
        raise ValueError(
            f"Checkpoint must contain exactly month {warmup_months}; "
            f"found state.month={state.month}, history={len(state.history)}"
        )
    expected_population = int(config["simulation"]["population"])
    expected_seed = int(config["simulation"]["seed"])
    if len(state.residents) != expected_population:
        raise ValueError(
            f"Checkpoint population {len(state.residents)} != requested {expected_population}"
        )
    if state.seed != expected_seed:
        raise ValueError(f"Checkpoint seed {state.seed} != requested {expected_seed}")
    state.scenario = scenario
    state.intents.clear()
    for row in state.history:
        row["scenario"] = scenario
    lineage = {
        "checkpoint": str(checkpoint_path),
        "origin_scenario": execution.get("scenario"),
        "origin_source_fingerprint": origin_source,
        "pre_equilibrium_config_fingerprint": current_pre_config,
        "cognitive_activation": cognitive_activation,
    }
    return state, audit, lineage


def _aggregate_decision_audits(
    output_dir: Path,
    *,
    provider: str | None = None,
) -> dict:
    sources = sorted(
        path for path in output_dir.rglob("decision_audit.jsonl")
        if path.parent != output_dir
    )
    target = output_dir / "decision_audit.jsonl"
    counts = {
        "records": 0,
        "accepted": 0,
        "fallbacks": 0,
        "bounded": 0,
        "rule_only": 0,
        "inactive": 0,
        "llm_eligible_records": 0,
        "unknown_status": 0,
        "sources": len(sources),
        "by_role": {},
        "fallback_categories": {},
        "model_provenance": {
            "successful_records": 0,
            "records_with_response_model": 0,
            "missing_response_model": 0,
            "exact_response_model": 0,
            "aliased_response_model": 0,
            "mismatched_response_model": 0,
            "pairs": {},
            "matching_policy": response_model_matching_policy(provider),
        },
    }
    with target.open("w", encoding="utf-8") as destination:
        for source in sources:
            for line in source.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                destination.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                counts["records"] += 1
                status = str(record.get("status", "unknown"))
                status_key = {
                    "accepted": "accepted",
                    "fallback": "fallbacks",
                    "bounded": "bounded",
                    "rule_only": "rule_only",
                    "inactive": "inactive",
                }.get(status, "unknown_status")
                counts[status_key] += 1
                llm_enabled = record.get("llm_enabled")
                if llm_enabled is None:
                    llm_enabled = status in {"accepted", "fallback", "bounded"}
                if bool(llm_enabled) and status != "inactive":
                    counts["llm_eligible_records"] += 1
                role = str(record.get("role", "unknown"))
                role_counts = counts["by_role"].setdefault(
                    role,
                    {
                        "records": 0,
                        "accepted": 0,
                        "fallbacks": 0,
                        "bounded": 0,
                        "rule_only": 0,
                        "inactive": 0,
                        "unknown_status": 0,
                        "llm_eligible_records": 0,
                    },
                )
                role_counts["records"] += 1
                role_counts[status_key] += 1
                if bool(llm_enabled) and status != "inactive":
                    role_counts["llm_eligible_records"] += 1
                category = record.get("fallback_category")
                if category:
                    category = str(category)
                    counts["fallback_categories"][category] = (
                        counts["fallback_categories"].get(category, 0) + 1
                    )
                if bool(llm_enabled) and status in {"accepted", "bounded"}:
                    model_audit = counts["model_provenance"]
                    model_audit["successful_records"] += 1
                    provenance = record.get("response_provenance") or {}
                    requested = provenance.get("requested_model")
                    response_model = provenance.get("response_model")
                    if response_model is None:
                        model_audit["missing_response_model"] += 1
                    else:
                        model_audit["records_with_response_model"] += 1
                    if requested is not None and response_model is not None:
                        requested_text = str(requested).removeprefix("openai/")
                        response_text = str(response_model).removeprefix("openai/")
                        pair_key = f"{requested_text} -> {response_text}"
                        model_audit["pairs"][pair_key] = (
                            model_audit["pairs"].get(pair_key, 0) + 1
                        )
                        match_kind = response_model_match_kind(
                            provider,
                            requested_text,
                            response_text,
                        )
                        if match_kind == "exact":
                            model_audit["exact_response_model"] += 1
                        elif match_kind == "alias":
                            model_audit["aliased_response_model"] += 1
                        else:
                            model_audit["mismatched_response_model"] += 1
    eligible = int(counts["llm_eligible_records"])
    counts["fallback_rate"] = (
        int(counts["fallbacks"]) / eligible if eligible else 0.0
    )
    for role_counts in counts["by_role"].values():
        role_eligible = int(role_counts["llm_eligible_records"])
        role_counts["fallback_rate"] = (
            int(role_counts["fallbacks"]) / role_eligible
            if role_eligible
            else 0.0
        )
    return counts


def _decision_quality_gate(
    decision_audit: dict,
    *,
    max_fallback_rate: float,
    max_role_fallback_rate: float,
    require_response_model_match: bool,
) -> dict:
    if not 0.0 <= max_fallback_rate <= 1.0:
        raise ValueError("--max-fallback-rate must be inside [0, 1]")
    if not 0.0 <= max_role_fallback_rate <= 1.0:
        raise ValueError("--max-role-fallback-rate must be inside [0, 1]")
    violations: list[str] = []
    overall = float(decision_audit.get("fallback_rate", 0.0))
    if overall > max_fallback_rate:
        violations.append(
            f"overall fallback rate {overall:.4%} exceeds {max_fallback_rate:.4%}"
        )
    role_rates = {}
    for role, row in sorted(decision_audit.get("by_role", {}).items()):
        if not int(row.get("llm_eligible_records", 0)):
            continue
        rate = float(row.get("fallback_rate", 0.0))
        role_rates[role] = rate
        if rate > max_role_fallback_rate:
            violations.append(
                f"{role} fallback rate {rate:.4%} exceeds "
                f"{max_role_fallback_rate:.4%}"
            )
    model_audit = decision_audit.get("model_provenance", {})
    if require_response_model_match:
        missing = int(model_audit.get("missing_response_model", 0))
        mismatched = int(model_audit.get("mismatched_response_model", 0))
        if missing:
            violations.append(
                f"{missing} successful LLM records lack a response model identifier"
            )
        if mismatched:
            violations.append(
                f"{mismatched} successful LLM records report a different response model"
            )
    return {
        "pass": not violations,
        "max_fallback_rate": max_fallback_rate,
        "max_role_fallback_rate": max_role_fallback_rate,
        "require_response_model_match": require_response_model_match,
        "observed_fallback_rate": overall,
        "observed_role_fallback_rates": role_rates,
        "violations": violations,
    }


def _reserve_future_firm_agent_specs(
    specs: list[dict],
    state: EconomyState,
    llm_roles: set[str],
    config: dict,
    *,
    total_months: int,
) -> tuple[list[dict], dict[str, int | None]]:
    """Pre-create immutable AgentSociety slots for every possible future entrant.

    AgentSociety 2 keeps its agent roster immutable after initialization, while
    the economic core can create and remove firms after the AI shock.  A
    conservative upper bound of ``remaining_steps * max_monthly_entries`` keeps
    those lifecycle changes inside the pre-created roster.  Inactive slots do
    not call the LLM and do not submit an economic intent.
    """
    firms_cfg = config.get("firms", {})
    remaining_steps = max(int(total_months) - int(state.month), 0)
    max_monthly_entries = (
        max(int(firms_cfg.get("max_monthly_entries", 1)), 0)
        if bool(firms_cfg.get("enable_entry_exit", False))
        else 0
    )
    reserved_count = remaining_steps * max_monthly_entries
    first_reserved_id = int(state.next_firm_id) if reserved_count else None
    last_reserved_id = (
        int(state.next_firm_id) + reserved_count - 1 if reserved_count else None
    )
    existing_ids = {int(spec["id"]) for spec in specs}
    for firm_id in range(
        int(state.next_firm_id),
        int(state.next_firm_id) + reserved_count,
    ):
        if firm_id in existing_ids:
            continue
        specs.append(
            {
                "id": firm_id,
                "profile": {
                    "id": firm_id,
                    "name": f"Firm-{firm_id}",
                    "role": "firm",
                    "economic_id": firm_id,
                },
                "config": {
                    "llm_enabled": "firm" in llm_roles,
                    "lifecycle_slot": True,
                },
            }
        )
    return specs, {
        "reserved_count": reserved_count,
        "first_reserved_id": first_reserved_id,
        "last_reserved_id": last_reserved_id,
    }


async def run_agentsociety(args: argparse.Namespace) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    provider_info = configure_agentsociety(
        args.provider, key_env=args.key_env, api_base=args.api_base, model=args.model
    )
    llm_runtime = _configure_llm_runtime(args)
    os.environ["WORKSPACE_PATH"] = str(PACKAGE_ROOT)

    from agentsociety2.config.llm_dispatcher import init_dispatchers
    from agentsociety2.registry import scan_and_register_custom_modules
    from agentsociety2.society import AgentSociety

    from .agent_society_runtime import create_environment_proxy

    checkpoint_path = getattr(args, "initial_state", None)
    checkpoint_raw = None
    if checkpoint_path is not None:
        checkpoint_raw = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
        checkpoint_population = len(checkpoint_raw["residents"])
        checkpoint_seed = int(checkpoint_raw["seed"])
        if args.population is not None and args.population != checkpoint_population:
            raise ValueError("--population conflicts with the checkpoint population")
        if args.seed is not None and args.seed != checkpoint_seed:
            raise ValueError("--seed conflicts with the checkpoint seed")
        population = checkpoint_population
        seed = checkpoint_seed
    else:
        population = args.population
        seed = args.seed
    config = scenario_config(
        load_config(args.config),
        args.scenario,
        population,
        seed,
        scenario_definition_version=getattr(
            args, "scenario_definition_version", None
        ),
    )
    if args.months is not None:
        config["simulation"]["months"] = args.months
    llm_roles = {item.strip() for item in args.llm_roles.split(",") if item.strip()}
    unknown_roles = llm_roles - {"resident", "firm", "government"}
    if unknown_roles:
        raise ValueError(f"Unknown LLM roles: {sorted(unknown_roles)}")
    if llm_roles and args.provider == "offline":
        raise ValueError("LLM roles require a non-offline provider")
    cognitive_regime = resolve_cognitive_regime(
        llm_roles, getattr(args, "cognitive_regime", None)
    )
    source_fingerprint = _source_fingerprint()
    checkpoint_audit: dict | None = None
    checkpoint_lineage: dict | None = None
    if checkpoint_path is not None:
        state, checkpoint_audit, checkpoint_lineage = _restore_pre_equilibrium_checkpoint(
            Path(checkpoint_path),
            config=config,
            scenario=args.scenario,
            llm_roles=llm_roles,
            provider_info=provider_info,
            source_fingerprint=source_fingerprint,
            allow_unstable=args.allow_unstable_equilibrium,
            allow_source_mismatch=getattr(args, "allow_checkpoint_source_mismatch", False),
            allow_cognitive_activation=getattr(
                args, "activate_cognitive_regime_from_checkpoint", False
            ),
        )
    else:
        state = initialize_economy(config)
    total_months = int(config["simulation"]["months"])
    starting_month = int(state.month)
    if starting_month > total_months:
        raise ValueError(
            f"Checkpoint month {starting_month} is later than requested final month {total_months}"
        )
    specs = build_agent_specs(state, llm_roles)
    specs, lifecycle_roster = _reserve_future_firm_agent_specs(
        specs,
        state,
        llm_roles,
        config,
        total_months=total_months,
    )
    batch_size = _positive_int(args.batch_size, "--batch-size")
    ray_task_count = math.ceil(len(specs) / batch_size)
    llm_runtime.update(
        {
            "batch_size": batch_size,
            "ray_task_count_per_month": ray_task_count,
            "initial_global_concurrency_ceiling": min(
                ray_task_count, llm_runtime["max_workers"]
            )
            * min(
                batch_size,
                llm_runtime["initial_concurrency_per_worker"],
            ),
        }
    )

    output_arg = getattr(args, "output", None)
    if output_arg is None:
        output_arg = matrix_cell_dir(
            root=getattr(args, "matrix_root", DEFAULT_MATRIX_ROOT),
            stage=getattr(args, "result_stage", "smoke"),
            scenario_definition_version=config["scenario_definition_version"],
            cognitive_regime=cognitive_regime,
            provider=args.provider,
            model=args.model,
            population=int(config["simulation"]["population"]),
            months=int(config["simulation"]["months"]),
            seed=int(config["simulation"]["seed"]),
        ) / args.scenario
    output_dir = Path(output_arg).resolve()
    if (output_dir / "SOCIETY.json").exists():
        raise FileExistsError(f"Run directory already contains a simulation: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        "config": config,
        "execution": {
            "scenario": args.scenario,
            "scenario_definition_version": config[
                "scenario_definition_version"
            ],
            "llm_roles": sorted(llm_roles),
            "provider": provider_info,
            "batch_size": batch_size,
            "llm_runtime": llm_runtime,
            "decision_quality_thresholds": {
                "max_fallback_rate": args.max_fallback_rate,
                "max_role_fallback_rate": args.max_role_fallback_rate,
                "require_response_model_match": args.require_response_model_match,
                "response_model_matching_policy": response_model_matching_policy(
                    args.provider
                ),
            },
            "firm_lifecycle_roster": lifecycle_roster,
            "source_fingerprint": source_fingerprint,
            "result_layout_version": RESULT_LAYOUT_VERSION,
            "result_stage": getattr(args, "result_stage", "smoke"),
            "cognitive_regime": cognitive_regime,
            "cognitive_activation_month": (
                int(config["simulation"]["shock_month"])
                if checkpoint_lineage
                and checkpoint_lineage.get("cognitive_activation")
                else None
            ),
            "matrix_cell": str(output_dir),
            "pre_equilibrium_config_fingerprint": _pre_equilibrium_config_fingerprint(config),
            "initial_state": str(Path(checkpoint_path).resolve()) if checkpoint_path else None,
            "checkpoint_lineage": checkpoint_lineage,
        },
    }
    resolved_config_path = output_dir / "resolved_config.json"
    _atomic_write_json(resolved_config_path, resolved)
    scan_result = scan_and_register_custom_modules(PACKAGE_ROOT)
    if scan_result.get("errors") or scan_result.get("registration_errors"):
        raise RuntimeError(f"Custom module registration failed: {scan_result}")

    # Enforce the configured Ray CPU ceiling before create_environment_proxy()
    # gets a chance to initialize Ray with all host CPUs.
    await init_dispatchers()
    proxy, actor = create_environment_proxy(state.to_dict(), config, output_dir)
    society = AgentSociety(
        agent_specs=specs,
        agent_class_name="EconomicAgent",
        env_router=proxy,
        start_t=datetime(2026, 1, 1) + timedelta(days=30 * state.month),
        run_dir=output_dir,
        batch_size=batch_size,
        enable_replay=args.replay,
    )
    await society.init()
    warmup_months = int(config["simulation"]["warmup_months"])
    pre_equilibrium_months = min(warmup_months, total_months)
    pre_equilibrium_audit: dict[str, object] | None = checkpoint_audit
    equilibrium_failure: str | None = None
    history: list[dict] = []
    token_stats: dict = {}
    try:
        if checkpoint_path is not None:
            # The checkpoint is already the qualified month-24 common state.
            # Run only the post-equilibrium months for this branch.
            if total_months > starting_month:
                await society.run(
                    num_steps=total_months - starting_month,
                    tick=30 * 24 * 3600,
                )
            history = await actor.get_history.remote()
        else:
            # LLM agents remain active throughout this stage.  It is an empirical
            # equilibrium qualification, not a deterministic burn-in period.
            await society.run(num_steps=pre_equilibrium_months, tick=30 * 24 * 3600)
            history = await actor.get_history.remote()
        if checkpoint_path is None and total_months >= warmup_months:
            pre_equilibrium_audit = audit_history(
                history,
                population=len(state.residents),
                seed=int(config["simulation"]["seed"]),
                scenario=args.scenario,
                warmup_months=warmup_months,
                thresholds=gate_thresholds(config),
                equilibrium_reference={
                    "employment_rate": sum(
                        resident.employed for resident in state.residents.values()
                    ) / len(state.residents),
                    "real_consumption": state.baseline_household_demand,
                    "aggregate_price": 1.0,
                    "firm_sales": state.baseline_total_output,
                },
            )
            (output_dir / "pre_equilibrium_audit.json").write_text(
                json.dumps(pre_equilibrium_audit, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            checkpoint = await actor.get_state.remote()
            (output_dir / "pre_equilibrium_state.json").write_text(
                json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if not bool(pre_equilibrium_audit["path_gate_pass"]) and not args.allow_unstable_equilibrium:
                equilibrium_failure = (
                    "Pre-equilibrium social-state gate failed at month "
                    f"{warmup_months}; post-shock execution was not started. "
                    "Inspect pre_equilibrium_audit.json and decision_audit.jsonl."
                )
            elif total_months > pre_equilibrium_months:
                await society.run(
                    num_steps=total_months - pre_equilibrium_months,
                    tick=30 * 24 * 3600,
                )
                history = await actor.get_history.remote()
        token_stats = dict(society._token_stats)
    finally:
        await society.close()
    for row in history:
        row["month"] = int(row["month"])
    write_history(history, output_dir)
    if checkpoint_path is not None:
        (output_dir / "pre_equilibrium_audit.json").write_text(
            json.dumps(pre_equilibrium_audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "pre_equilibrium_state.json").write_text(
            json.dumps(checkpoint_raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        source_decisions = Path(checkpoint_path).resolve().parent / "decision_audit.jsonl"
        if source_decisions.is_file():
            shutil.copyfile(source_decisions, output_dir / "pre_equilibrium_decision_audit.jsonl")
    decision_audit = _aggregate_decision_audits(
        output_dir,
        provider=args.provider,
    )
    behavior_audit = write_behavior_audit(
        output_dir / "decision_audit.jsonl", output_dir
    )
    decision_audit["behavior_summary_records"] = int(behavior_audit["records"])
    decision_audit["behavior_summary_artifacts"] = dict(
        behavior_audit["artifacts"]
    )
    decision_audit["behavior_qualification"] = dict(
        behavior_audit["behavior_qualification"]
    )
    executed_history_months = len(history) - starting_month
    decision_audit["expected_records"] = len(specs) * executed_history_months
    decision_audit["expected_llm_records"] = decision_audit[
        "llm_eligible_records"
    ]
    decision_audit["llm_records"] = (
        decision_audit["accepted"] + decision_audit["fallbacks"] + decision_audit["bounded"]
    )
    decision_audit["closed"] = bool(
        decision_audit["records"] == decision_audit["expected_records"]
        and decision_audit["llm_records"] == decision_audit["expected_llm_records"]
        and decision_audit["unknown_status"] == 0
    )
    audit_failure = None if decision_audit["closed"] else (
        "Decision audit did not close: "
        f"records={decision_audit['records']}/{decision_audit['expected_records']}, "
        f"llm_records={decision_audit['llm_records']}/{decision_audit['expected_llm_records']}, "
        f"unknown_status={decision_audit['unknown_status']}"
    )
    decision_audit["quality_gate"] = _decision_quality_gate(
        decision_audit,
        max_fallback_rate=float(args.max_fallback_rate),
        max_role_fallback_rate=float(args.max_role_fallback_rate),
        require_response_model_match=bool(args.require_response_model_match),
    )
    quality_failure = (
        None
        if decision_audit["quality_gate"]["pass"]
        else "Decision quality gate failed: "
        + "; ".join(decision_audit["quality_gate"]["violations"])
    )
    status = (
        "equilibrium_failed" if equilibrium_failure
        else "audit_failed" if audit_failure
        else "quality_gate_failed" if quality_failure
        else "completed"
    )
    terminal_error = equilibrium_failure or audit_failure or quality_failure
    completed_steps = max(len(history) - starting_month, 0)
    step_metadata = _finalize_step_metadata(
        output_dir,
        status=status,
        requested_steps=max(total_months - starting_month, 0),
        completed_steps=completed_steps,
        final_model_month=len(history),
    )
    run_manifest = _write_run_manifest(
        output_dir,
        status=status,
        started_at=started_at,
        source_fingerprint=source_fingerprint,
        resolved_config_path=resolved_config_path,
        scenario=args.scenario,
        scenario_definition_version=config["scenario_definition_version"],
        population=len(state.residents),
        seed=int(config["simulation"]["seed"]),
        starting_month=starting_month,
        requested_final_month=total_months,
        completed_history_months=len(history),
        completed_steps=completed_steps,
        decision_audit=decision_audit,
        error=terminal_error,
    )
    result = {
        "status": status,
        "scenario": args.scenario,
        "scenario_definition_version": config[
            "scenario_definition_version"
        ],
        "population": len(state.residents),
        "firms": len(state.firms),
        "agents": len(specs),
        "months": len(history),
        "starting_month": starting_month,
        "executed_months": completed_steps,
        "checkpoint_lineage": checkpoint_lineage,
        "llm_roles": sorted(llm_roles),
        "provider": provider_info,
        "token_stats": token_stats,
        "source_fingerprint": source_fingerprint,
        "pre_equilibrium_audit": pre_equilibrium_audit,
        "decision_audit": decision_audit,
        "behavior_audit": {
            "records": behavior_audit["records"],
            "by_role": behavior_audit["by_role"],
            "by_status": behavior_audit["by_status"],
            "action_distributions": behavior_audit["action_distributions"],
            "monotonicity_checks": behavior_audit["monotonicity_checks"],
            "artifacts": behavior_audit["artifacts"],
        },
        "step_metadata": step_metadata,
        "run_manifest": run_manifest,
        "summary": summarize(history, int(config["simulation"]["shock_month"])),
    }
    _atomic_write_json(output_dir / "summary.json", result)
    if equilibrium_failure:
        raise RuntimeError(equilibrium_failure)
    if audit_failure:
        raise RuntimeError(audit_failure)
    if quality_failure:
        raise RuntimeError(quality_failure)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AgentSociety AI-employment execution model")
    parser.add_argument("--scenario", choices=[f"E{i}" for i in range(7)], default="E5")
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--months", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--scenario-definition-version",
        default=None,
        help=(
            "Select versioned E5/E6 semantics; the baseline default is "
            "institutional_v2, while legacy_v1 reproduces historical definitions"
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
    parser.add_argument(
        "--llm-max-workers",
        type=int,
        default=None,
        help="Maximum concurrent Ray agent-batch workers; defaults to AGENTSOCIETY_LLM_RAY_MAX_WORKERS or host CPUs",
    )
    parser.add_argument(
        "--llm-concurrency",
        type=int,
        default=None,
        help="Initial LLM request concurrency inside each Ray worker; defaults to AGENTSOCIETY_LLM_RAY_CONCURRENCY or 16",
    )
    parser.add_argument(
        "--max-fallback-rate",
        type=float,
        default=0.01,
        help="Fail the decision quality gate when the overall LLM fallback rate exceeds this fraction",
    )
    parser.add_argument(
        "--max-role-fallback-rate",
        type=float,
        default=0.01,
        help="Fail the decision quality gate when any LLM role fallback rate exceeds this fraction",
    )
    parser.add_argument(
        "--require-response-model-match",
        action="store_true",
        help="Fail the decision quality gate unless every successful response reports the requested model",
    )
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--provider", choices=["offline", "hkust", "deepseek", "dashscope", "moonshot", "openai", "custom"], default="offline")
    parser.add_argument("--llm-roles", default="", help="Comma-separated: resident,firm,government")
    parser.add_argument("--key-env", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--initial-state",
        type=Path,
        default=None,
        help="Qualified pre_equilibrium_state.json to branch from at month 24",
    )
    parser.add_argument(
        "--allow-unstable-equilibrium",
        action="store_true",
        help="Continue after a failed month-24 equilibrium gate (diagnostic use only)",
    )
    parser.add_argument(
        "--allow-checkpoint-source-mismatch",
        action="store_true",
        help="Load a checkpoint produced by different source code (diagnostic use only)",
    )
    parser.add_argument(
        "--activate-cognitive-regime-from-checkpoint",
        action="store_true",
        help=(
            "Allow an R1/R2/R3 branch to start from a qualified offline R0 "
            "month-24 checkpoint; roles activate at the month-25 shock"
        ),
    )
    return parser


def main() -> None:
    result = asyncio.run(run_agentsociety(build_parser().parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
