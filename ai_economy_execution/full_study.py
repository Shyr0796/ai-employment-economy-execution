from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .configuration import load_config
from .experiments import parse_integer_set, run_numeric_experiment
from .result_layout import (
    DEFAULT_MATRIX_ROOT,
    VALID_RESULT_STAGES,
    matrix_aggregate_dir,
)
from .sensitivity import run_sensitivity_analysis


PACKAGE_ROOT = Path(__file__).resolve().parent


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        digest.update(path.relative_to(PACKAGE_ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_full_report(output: Path, main_result: dict[str, Any], sensitivity_result: dict[str, Any] | None) -> None:
    lines = [
        "# AI就业与需求反馈模型：完整自动研究",
        "",
        "## 已执行模块",
        "",
        f"- 主实验路径：{len(main_result['runs'])}",
        f"- 种子内情景比较：{len(main_result['comparisons'])}",
        f"- Bootstrap重复次数：{main_result['bootstrap']['samples']}",
        f"- 正式运行门槛：{'通过' if main_result['run_gates']['passed'] else '未通过'}",
        "- 月度分布：总体、五收入组、底部60%、底部80%",
        "- 不平等指标：收入与实际消费Atkinson ε=0.5/1.0/1.5",
    ]
    if sensitivity_result is not None:
        lines.append(f"- 单因素敏感性结果行：{len(sensitivity_result['effects'])}")
    else:
        lines.append("- 单因素敏感性分析：本次跳过")
    lines.extend([
        "",
        "## 报告入口",
        "",
        "- [主实验报告](main_experiment/research_report.md)",
    ])
    if sensitivity_result is not None:
        lines.append("- [敏感性分析报告](sensitivity/sensitivity_report.md)")
    lines.extend([
        "",
        "所有结论必须结合主实验Bootstrap区间、不同人口规模和敏感性边界共同解释。",
        "",
    ])
    (output / "FULL_STUDY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_full_study(
    config_path: Path | None,
    populations: list[int],
    seeds: list[int],
    output: Path,
    *,
    bootstrap_samples: int = 5000,
    bootstrap_confidence: float = 0.95,
    include_sensitivity: bool = True,
    scenario_definition_version: str | None = None,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Study output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    manifest = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": _source_hash(),
        "config_path": str(config_path) if config_path else "package-default",
        "populations": populations,
        "seeds": seeds,
        "scenarios": list(config["scenarios"]),
        "scenario_definition_version": (
            scenario_definition_version
            or config.get("default_scenario_definition_version", "unversioned")
        ),
        "months": int(config["simulation"]["months"]),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_confidence": bootstrap_confidence,
        "sensitivity": include_sensitivity,
        "api_mode": "offline-numeric-core",
    }
    manifest_path = output / "study_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        main_result = run_numeric_experiment(
            config_path,
            populations,
            seeds,
            output / "main_experiment",
            baseline_config=config,
            write_paths=True,
            bootstrap_samples=bootstrap_samples,
            bootstrap_confidence=bootstrap_confidence,
            generate_artifacts=True,
            scenario_definition_version=scenario_definition_version,
        )
        manifest["run_gates_passed"] = bool(main_result["run_gates"]["passed"])
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if not main_result["run_gates"]["passed"]:
            raise RuntimeError(
                "Formal run gates failed; inspect main_experiment/run_gate_audit.csv before sensitivity analysis"
            )
        sensitivity_result = None
        if include_sensitivity:
            sensitivity_result = run_sensitivity_analysis(
                config_path,
                populations,
                seeds,
                output / "sensitivity",
                baseline_config=config,
                baseline_result=main_result,
                bootstrap_samples=bootstrap_samples,
                bootstrap_confidence=bootstrap_confidence,
                scenario_definition_version=scenario_definition_version,
            )
        _write_full_report(output, main_result, sensitivity_result)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    manifest["status"] = "complete"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["main_runs"] = len(main_result["runs"])
    manifest["sensitivity_effect_rows"] = len(sensitivity_result["effects"]) if sensitivity_result else 0
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest": manifest, "main": main_result, "sensitivity": sensitivity_result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete offline confirmatory study and sensitivity pipeline")
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
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--skip-sensitivity", action="store_true")
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
        analysis="full_study",
        populations=args.populations,
        seeds=args.seeds,
    )
    result = run_full_study(
        args.config,
        parse_integer_set(args.populations),
        parse_integer_set(args.seeds),
        output,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_confidence=args.bootstrap_confidence,
        include_sensitivity=not args.skip_sensitivity,
        scenario_definition_version=args.scenario_definition_version,
    )
    print(json.dumps(result["manifest"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
