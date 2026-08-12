"""Writers for simulation records, summaries, and research reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from ..metrics import ATKINSON_EPSILONS, INCOME_GROUPS


def write_records_csv(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    rows = list(records)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_aggregate(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for population, scenarios in aggregate.items():
        for scenario, metrics in scenarios.items():
            for metric, statistics in metrics.items():
                if not isinstance(statistics, dict):
                    continue
                rows.append({"population": int(population), "scenario": scenario, "metric": metric, **statistics})
    return rows


def flatten_run_gates(run_gates: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, gate_type in (
        ("path_audits", "path"),
        ("counterfactual_audits", "counterfactual"),
        ("systematic_pretrend_audits", "systematic_pretrend"),
    ):
        rows.extend({"gate_type": gate_type, **row} for row in run_gates.get(key, []))
    return rows


def _number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _percent(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * float(value):.2f}%"


def write_research_report(result: dict[str, Any], output_dir: str | Path, config: dict[str, Any]) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    populations = sorted(int(value) for value in result["aggregate"])
    scenarios = list(config["scenarios"])
    seeds = sorted({int(row["seed"]) for row in result["runs"]})
    bootstrap = result.get("bootstrap", {})
    lines = [
        "# AI就业与需求反馈模型：自动研究报告",
        "",
        "## 实验设计",
        "",
        f"- 人口规模：{', '.join(str(value) for value in populations)}",
        f"- 随机种子：{len(seeds)}组（{seeds[0] if seeds else 'NA'}—{seeds[-1] if seeds else 'NA'}）",
        f"- 情景：{', '.join(scenarios)}",
        f"- 每条路径：{config['simulation']['months']}个月；AI冲击从第{config['simulation']['shock_month']}月开始",
        f"- 配对Bootstrap：{bootstrap.get('samples', 'NA')}次，置信水平{_percent(bootstrap.get('confidence'))}",
        f"- 正式运行门槛：{'通过' if result.get('run_gates', {}).get('passed') else '未通过'}",
        "",
        "## 首要结果：底部60%累计实际消费",
        "",
        "|人口|情景|比较基准|均值|中位数|Q1|Q3|Bootstrap区间|非负种子占比|居民获益通过率|",
        "|---:|---|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for population in populations:
        for scenario in scenarios:
            metrics = result["aggregate"][str(population)][scenario]
            primary = metrics.get("bottom60_cumulative_real_consumption_gain", {})
            benefit = metrics.get("ordinary_resident_benefit", {})
            control = "E0" if scenario in {"E0", "E1"} else "E1"
            interval = f"[{_number(primary.get('bootstrap_ci_low'))}, {_number(primary.get('bootstrap_ci_high'))}]"
            lines.append(
                f"|{population}|{scenario}|{control}|{_number(primary.get('mean'))}|{_number(primary.get('median'))}|"
                f"{_number(primary.get('q1'))}|{_number(primary.get('q3'))}|{interval}|"
                f"{_percent(primary.get('positive_seed_share'))}|{_percent(benefit.get('pass_seed_share'))}|"
            )

    lines.extend([
        "",
        "## 分组累计实际消费变化",
        "",
        "|人口|情景|低收入|中间偏下|中间|中间偏上|高收入|底部80%|",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for population in populations:
        for scenario in scenarios:
            metrics = result["aggregate"][str(population)][scenario]
            values = [
                _percent(metrics.get(f"group_{group}_cumulative_real_consumption_gain", {}).get("mean"))
                for group in INCOME_GROUPS
            ]
            bottom80 = _percent(metrics.get("bottom80_cumulative_real_consumption_gain", {}).get("mean"))
            lines.append(f"|{population}|{scenario}|{'|'.join(values)}|{bottom80}|")

    lines.extend([
        "",
        "## 不平等结果：末12个月Atkinson指数相对变化",
        "",
        "负值表示相对比较基准的不平等下降，正值表示不平等上升。",
        "",
        "|人口|情景|收入 ε=0.5|收入 ε=1.0|收入 ε=1.5|消费 ε=0.5|消费 ε=1.0|消费 ε=1.5|",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for population in populations:
        for scenario in scenarios:
            metrics = result["aggregate"][str(population)][scenario]
            income = [
                _number(metrics.get(f"tail_disposable_income_atkinson_{str(epsilon).replace('.', '_')}_delta", {}).get("mean"), 6)
                for epsilon in ATKINSON_EPSILONS
            ]
            consumption = [
                _number(metrics.get(f"tail_real_consumption_atkinson_{str(epsilon).replace('.', '_')}_delta", {}).get("mean"), 6)
                for epsilon in ATKINSON_EPSILONS
            ]
            lines.append(f"|{population}|{scenario}|{'|'.join(income + consumption)}|")

    lines.extend([
        "",
        "## 输出说明",
        "",
        "- `runs.csv`：每条路径的终期与累计摘要。",
        "- `comparisons.csv`：种子内配对情景差。",
        "- `aggregate_statistics.csv`：均值、中位数、Q1、Q3、IQR与Bootstrap区间。",
        "- `run_gate_audit.csv`：会计、边界、热身稳定性、初始一致性与冲击前可比性审计。",
        "- 各路径目录中的 `metrics.csv/json`：完整月度轨迹和分组指标。",
        "- Atkinson指数分别使用0.5、1.0和1.5三种不平等厌恶参数，不能跨参数直接比较水平。",
        "",
    ])
    report_path = output / "research_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_experiment_artifacts(result: dict[str, Any], output_dir: str | Path, config: dict[str, Any]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "resolved_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_records_csv(result["runs"], output / "runs.csv")
    write_records_csv(result["comparisons"], output / "comparisons.csv")
    write_records_csv(result["policy_interactions"], output / "policy_interactions.csv")
    write_records_csv(flatten_aggregate(result["aggregate"]), output / "aggregate_statistics.csv")
    write_records_csv(flatten_run_gates(result.get("run_gates", {})), output / "run_gate_audit.csv")
    write_research_report(result, output, config)
