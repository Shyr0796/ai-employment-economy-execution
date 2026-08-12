from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


RESULT_LAYOUT_VERSION = "matrix_v1"
DEFAULT_MATRIX_ROOT = Path("ai_economy_execution/results/research_matrix")

COGNITIVE_REGIME_ROLES: dict[str, tuple[str, ...]] = {
    "R0": (),
    "R1": ("government",),
    "R2": ("firm", "government"),
    "R3": ("resident", "firm", "government"),
}

COGNITIVE_REGIME_DIRS = {
    "R0": "R0_rules",
    "R1": "R1_government",
    "R2": "R2_firm_government",
    "R3": "R3_resident_firm_government",
}

VALID_RESULT_STAGES = ("smoke", "pilot", "formal")


def normalize_roles(roles: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(roles, str):
        values = roles.split(",")
    else:
        values = roles
    normalized = tuple(
        sorted({str(value).strip().lower() for value in values if str(value).strip()})
    )
    unknown = set(normalized) - {"resident", "firm", "government"}
    if unknown:
        raise ValueError(f"Unknown LLM roles: {sorted(unknown)}")
    return normalized


def resolve_cognitive_regime(
    roles: str | Iterable[str], explicit_regime: str | None = None
) -> str:
    normalized = normalize_roles(roles)
    inferred = next(
        (
            regime
            for regime, expected_roles in COGNITIVE_REGIME_ROLES.items()
            if tuple(sorted(expected_roles)) == normalized
        ),
        None,
    )
    if inferred is None:
        raise ValueError(
            "LLM roles must match one registered cognitive regime: "
            "R0='', R1='government', R2='firm,government', or "
            "R3='resident,firm,government'"
        )
    if explicit_regime is not None:
        requested = explicit_regime.strip().upper()
        if requested not in COGNITIVE_REGIME_ROLES:
            raise ValueError(f"Unknown cognitive regime: {explicit_regime}")
        if requested != inferred:
            raise ValueError(
                f"Cognitive regime {requested} conflicts with LLM roles "
                f"{list(normalized)}; inferred {inferred}"
            )
    return inferred


def _slug(value: str | None, fallback: str) -> str:
    text = (value or fallback).strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "-", text).strip("-")
    return text or fallback


def provider_model_dir(provider: str, model: str | None) -> str:
    provider_slug = _slug(provider, "offline")
    if provider_slug == "offline":
        return "offline_rules"
    return f"{provider_slug}_{_slug(model, 'default-model')}"


def matrix_cell_dir(
    *,
    root: Path = DEFAULT_MATRIX_ROOT,
    stage: str,
    scenario_definition_version: str,
    cognitive_regime: str,
    provider: str,
    model: str | None,
    population: int,
    months: int,
    seed: int,
) -> Path:
    normalized_stage = stage.strip().lower()
    if normalized_stage not in VALID_RESULT_STAGES:
        raise ValueError(
            f"Unknown result stage {stage!r}; choose from {VALID_RESULT_STAGES}"
        )
    regime = cognitive_regime.strip().upper()
    if regime not in COGNITIVE_REGIME_DIRS:
        raise ValueError(f"Unknown cognitive regime: {cognitive_regime}")
    if population <= 0 or months <= 0 or seed < 0:
        raise ValueError("Population/months must be positive and seed non-negative")
    return (
        Path(root)
        / normalized_stage
        / _slug(scenario_definition_version, "unversioned")
        / COGNITIVE_REGIME_DIRS[regime]
        / provider_model_dir(provider, model)
        / f"N{population:05d}_M{months:03d}_S{seed:03d}"
    )


def matrix_aggregate_dir(
    *,
    root: Path = DEFAULT_MATRIX_ROOT,
    stage: str,
    scenario_definition_version: str,
    cognitive_regime: str,
    provider: str,
    model: str | None,
    analysis: str,
    populations: str,
    seeds: str,
) -> Path:
    base = matrix_cell_dir(
        root=root,
        stage=stage,
        scenario_definition_version=scenario_definition_version,
        cognitive_regime=cognitive_regime,
        provider=provider,
        model=model,
        population=1,
        months=1,
        seed=0,
    ).parent
    return (
        base
        / "aggregate"
        / (
            f"{_slug(analysis, 'analysis')}"
            f"__P{_slug(populations, 'unspecified')}"
            f"__S{_slug(seeds, 'unspecified')}"
        )
    )
