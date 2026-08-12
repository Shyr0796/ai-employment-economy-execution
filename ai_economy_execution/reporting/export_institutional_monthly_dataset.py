#!/usr/bin/env python3
"""Export completed institutional_v2 R1/R2 runs to one plotting-ready Parquet file."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
RESULTS = (
    ROOT
    / "ai_economy_execution"
    / "results"
    / "research_matrix"
    / "formal"
    / "institutional_v2"
)
OUTPUT_DIR = ROOT / "output" / "research_data"
DATA_PATH = OUTPUT_DIR / "institutional_v2_R1_R2_E0_E6_monthly.parquet"
DOC_PATH = OUTPUT_DIR / "institutional_v2_R1_R2_E0_E6_data_dictionary.md"
ACTIVATION_MONTH = 25

REGIMES = {
    "R1_government": {
        "code": "R1",
        "name": "Government LLM",
        "definition": "Government is LLM-driven; firms and residents are rule-driven.",
    },
    "R2_firm_government": {
        "code": "R2",
        "name": "Firm + Government LLM",
        "definition": "Firms and government are LLM-driven; residents are rule-driven.",
    },
}

SCENARIOS = {
    "E0": ("No New AI", "No new private AI and no new intervention; time counterfactual."),
    "E1": ("Laissez-Faire Private AI Adoption", "Private AI adoption without a dedicated social-return policy."),
    "E2": (
        "Employment-Preserving AI Responsibility with Cost Sharing",
        "Employment responsibility, shorter hours, wage-cost sharing, restructuring grace, and distress exemptions.",
    ),
    "E3": (
        "AI Infrastructure Levy and Social Return",
        "AI productivity-rent levy earmarked for public services and public investment.",
    ),
    "E4": (
        "AI Time Dividend and Solo-Enterprise Formation",
        "Personal AI and solo-enterprise formation with explicit new-demand and incumbent-displacement channels.",
    ),
    "E5": (
        "Integrated AI Social Compact",
        "E2, E3, and E4 combined with active transfers, procurement, and government AI.",
    ),
    "E6": (
        "Integrated AI Social Compact under Fiscal Constraints",
        "E5 under tighter annual-deficit and debt limits.",
    ),
}

PROVENANCE_COLUMNS = [
    "dataset_schema_version",
    "study_stage",
    "scenario_definition_version",
    "regime_code",
    "regime_name",
    "regime_definition",
    "scenario_code",
    "scenario_name",
    "scenario_definition",
    "run_id",
    "seed",
    "population",
    "configured_months",
    "activation_month",
    "phase",
    "months_since_activation",
    "policy_active",
    "provider",
    "requested_model",
    "api_base",
    "llm_roles_json",
    "response_model_pairs_json",
    "source_fingerprint",
    "run_status",
    "run_relative_path",
    "metrics_source_sha256",
    "summary_source_sha256",
    "decision_records",
    "decision_accepted",
    "decision_fallbacks",
    "decision_fallback_rate",
    "decision_llm_records",
    "decision_expected_llm_records",
    "behavior_suitable_for_claims",
    "behavior_qualification_json",
    "pre_equilibrium_accounting_pass",
    "pre_equilibrium_boundary_pass",
    "pre_equilibrium_warmup_stability_pass",
    "pre_equilibrium_path_gate_pass",
    "pre_equilibrium_max_accounting_absolute_error",
    "pre_equilibrium_max_accounting_tolerance_ratio",
]

CORE_DESCRIPTIONS = {
    "month": "模型月份，范围为1至configured_months。",
    "phase": "pre_equilibrium表示冲击前均衡期；post_shock表示制度与AI激活期。",
    "months_since_activation": "相对激活月的月份；激活月为0，激活前为负数。",
    "policy_active": "该月AI与制度处理是否已经激活。",
    "real_consumption": "当月总体实际消费。",
    "bottom60_real_consumption": "当月收入底部60%家庭实际消费总量。",
    "bottom80_real_consumption": "当月收入底部80%家庭实际消费总量。",
    "disposable_income": "当月居民可支配收入总量。",
    "employment_rate": "当月总体就业率。",
    "unemployment_rate": "当月总体失业率。",
    "wage_employment_rate": "当月工资就业率。",
    "self_employment_rate": "当月自雇率。",
    "long_unemployment_rate": "当月长期失业率。",
    "average_required_work_hours": "当月平均必要工作时数。",
    "average_work_intensity": "当月平均工作强度指数。",
    "ai_attributable_layoffs_blocked": "当月被就业责任机制阻止的AI归因裁员数。",
    "cumulative_ai_attributable_layoffs_blocked": "截至当月累计被阻止的AI归因裁员数。",
    "firm_exit_layoffs": "当月因企业退出造成的岗位损失。",
    "cumulative_firm_exit_layoffs": "截至当月累计企业退出岗位损失。",
    "firm_count": "当月存续企业数量。",
    "firm_entries": "当月企业进入数量。",
    "firm_exits": "当月企业退出数量。",
    "market_hhi": "当月市场赫芬达尔-赫希曼集中度指数。",
    "firm_price_dispersion": "当月企业价格离散度。",
    "below_cost_pricing_market_share": "当月低于成本定价企业的市场份额。",
    "unmet_final_demand": "当月未满足最终需求。",
    "capacity_utilization": "当月总体产能利用率。",
    "firm_sales": "当月企业销售总额。",
    "solo_entries": "当月个体企业进入数。",
    "solo_exits": "当月个体企业退出数。",
    "solo_enterprise_sales": "当月个体企业销售额。",
    "solo_enterprise_income": "当月个体企业经营收入。",
    "solo_net_additional_demand": "当月个体企业创造的净新增需求。",
    "solo_incumbent_displacement": "当月个体企业销售对在位企业的替代额。",
    "solo_b2b_sales": "当月个体企业B2B销售额。",
    "solo_external_sales": "当月个体企业外部市场销售额。",
    "public_service_index": "当月公共服务指数。",
    "ai_infrastructure_levy": "当月AI基础设施征费收入。",
    "government_ai_levy_fund_balance": "当月末AI征费专项基金余额。",
    "government_ai_levy_public_service_spending": "当月AI征费用于公共服务的支出。",
    "government_ai_levy_public_investment": "当月AI征费用于公共投资的支出。",
    "government_retention_wage_subsidy": "当月政府保就业工资补贴。",
    "government_fiscal_balance": "当月政府财政余额。",
    "government_debt_ratio": "当月政府总负债率。",
    "government_arrears": "当月政府欠款余额。",
    "government_fiscal_curtailment": "当月因财政约束被裁减的支出。",
    "government_statutory_funding_gap": "当月法定资金缺口。",
    "bank_firm_credit_requested": "当月企业申请的银行信贷。",
    "bank_firm_credit_rejected": "当月被拒绝的企业信贷。",
    "bank_capital_adequacy_ratio": "当月银行资本充足率。",
    "bank_npl_ratio": "当月银行不良贷款率。",
    "bank_reserve_ratio": "当月银行准备金率。",
    "aggregate_price": "当月总体价格指数。",
    "firm_average_price": "当月企业平均价格。",
    "personal_ai_mean_use_rate": "当月居民个人AI平均使用率。",
    "government_ai_use_rate": "当月政府AI使用率。",
    "disposable_income_atkinson_1_0": "当月可支配收入Atkinson指数，epsilon=1.0。",
    "real_consumption_atkinson_1_0": "当月实际消费Atkinson指数，epsilon=1.0。",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def classify_column(column: str) -> str:
    if column in PROVENANCE_COLUMNS:
        return "01 Provenance and validity"
    if column in {"month", "scenario", "population"}:
        return "02 Time and identity"
    if column.startswith(("bottom60_", "bottom80_", "group_")) or "atkinson" in column:
        return "09 Distribution and income groups"
    if column.startswith("bank_") or "credit_to_" in column or "credit_ratio" in column:
        return "08 Financial system"
    if column.startswith("solo_") or "voluntary_wage_exits" in column:
        return "06 Solo enterprise and new demand"
    if column.startswith("government_") or column in {
        "labor_tax",
        "indirect_tax",
        "profit_tax",
        "ai_infrastructure_levy",
        "ai_levy_rent_base",
        "ai_levy_capture_rate",
        "public_service_index",
    }:
        return "07 Government, levy, public service, and fiscal"
    if column.startswith("culture_"):
        return "05 Firm culture and strategy"
    if any(
        token in column
        for token in (
            "employment",
            "unemployment",
            "hires",
            "fires",
            "layoff",
            "work_intensity",
            "work_hours",
            "wage_bill",
            "restructuring",
        )
    ):
        return "04 Employment and work"
    if column.startswith("firm_") or any(
        token in column
        for token in (
            "market_hhi",
            "capacity",
            "output",
            "price",
            "retained_profit",
            "distressed_firm",
            "unsold",
            "entry_jobs",
            "exit_jobs",
        )
    ):
        return "05 Firms, production, prices, and market structure"
    if column.startswith("household_") or any(
        token in column
        for token in (
            "consumption",
            "disposable_income",
            "cash",
            "deposit",
            "financial_wealth",
            "managed_fund",
            "saving",
            "vulnerable",
            "stress",
        )
    ):
        return "03 Households, income, consumption, and resilience"
    return "10 Other macro and accounting metrics"


def infer_unit(column: str, dtype: Any) -> str:
    if column in PROVENANCE_COLUMNS:
        if str(dtype) == "bool":
            return "boolean"
        return "text / run-level value"
    if column == "month" or column.endswith("_months"):
        return "month"
    if column.endswith(("_rate", "_ratio", "_share", "_retention")):
        return "proportion [0,1]"
    if "atkinson" in column or column.endswith("_index") or "work_intensity" in column:
        return "dimensionless index"
    if column.endswith(("_count", "_entries", "_exits", "_hires", "_fires", "_layoffs")):
        return "count"
    if any(
        token in column
        for token in (
            "spending",
            "sales",
            "income",
            "revenue",
            "purchase",
            "procurement",
            "investment",
            "capital",
            "cash",
            "debt",
            "arrears",
            "tax",
            "profit",
            "wage",
            "assets",
            "loans",
            "funding",
            "credit",
            "output",
            "consumption",
            "shortfall",
            "curtailment",
        )
    ):
        return "model monetary unit"
    if "price" in column:
        return "price / price index"
    if "hours" in column:
        return "hours"
    if str(dtype).startswith(("int", "uint")):
        return "count / integer"
    return "model unit"


def describe_column(column: str) -> str:
    if column in CORE_DESCRIPTIONS:
        return CORE_DESCRIPTIONS[column]
    provenance = {
        "dataset_schema_version": "统一数据集结构版本。",
        "study_stage": "研究结果阶段，本文件固定为formal。",
        "scenario_definition_version": "E0-E6制度定义版本。",
        "regime_code": "认知架构代码R1或R2。",
        "regime_name": "认知架构名称。",
        "regime_definition": "哪些主体由LLM驱动。",
        "scenario_code": "制度情景代码E0-E6。",
        "scenario_name": "制度情景英文名称。",
        "scenario_definition": "制度情景简要定义。",
        "run_id": "运行唯一标识。",
        "seed": "随机种子。",
        "population": "居民主体数量。",
        "configured_months": "配置的总模拟月份。",
        "activation_month": "AI与制度处理统一激活月份。",
        "provider": "LLM接口提供方。",
        "requested_model": "运行请求的模型名称。",
        "api_base": "运行记录中的API基础地址。",
        "llm_roles_json": "本运行实际启用的LLM角色JSON。",
        "response_model_pairs_json": "请求模型与服务端响应模型映射JSON。",
        "source_fingerprint": "运行源码与配置指纹。",
        "run_status": "运行完成状态。",
        "run_relative_path": "相对项目根目录的原始结果目录。",
        "metrics_source_sha256": "原始metrics.csv的SHA-256。",
        "summary_source_sha256": "原始summary.json的SHA-256。",
        "decision_records": "逐决策审计记录数。",
        "decision_accepted": "被接受的LLM决策数。",
        "decision_fallbacks": "回退到规则的决策数。",
        "decision_fallback_rate": "决策回退比例。",
        "decision_llm_records": "实际LLM决策记录数。",
        "decision_expected_llm_records": "预期LLM决策记录数。",
        "behavior_suitable_for_claims": "行为资格是否支持正式行为结论。",
        "behavior_qualification_json": "完整行为资格检查JSON。",
        "pre_equilibrium_accounting_pass": "前置均衡会计门槛是否通过。",
        "pre_equilibrium_boundary_pass": "前置均衡数值边界门槛是否通过。",
        "pre_equilibrium_warmup_stability_pass": "前置均衡稳定性门槛是否通过。",
        "pre_equilibrium_path_gate_pass": "前置均衡综合路径门槛是否通过。",
        "pre_equilibrium_max_accounting_absolute_error": "前置均衡最大会计绝对误差。",
        "pre_equilibrium_max_accounting_tolerance_ratio": "最大会计误差与容许值之比。",
    }
    if column in provenance:
        return provenance[column]
    return "月度模型输出：" + column.replace("_", " ") + "。"


def discover_completed_runs() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for regime_dir, regime in REGIMES.items():
        base = RESULTS / regime_dir
        if not base.exists():
            continue
        for metrics_path in sorted(base.glob("*/N*_M*_S*/E[0-6]/metrics.csv")):
            run_dir = metrics_path.parent
            summary_path = run_dir / "summary.json"
            manifest_path = run_dir / "run_manifest.json"
            if not summary_path.exists() or not manifest_path.exists():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if summary.get("status") != "completed" or manifest.get("status") != "completed":
                continue
            scenario = summary["scenario"]
            if scenario not in SCENARIOS:
                continue
            design = summary["run_manifest"]["design"]
            records.append(
                {
                    "regime_dir": regime_dir,
                    "regime": regime,
                    "scenario": scenario,
                    "run_dir": run_dir,
                    "metrics_path": metrics_path,
                    "summary_path": summary_path,
                    "manifest_path": manifest_path,
                    "summary": summary,
                    "manifest": manifest,
                    "design": design,
                }
            )
    records.sort(
        key=lambda item: (
            item["regime"]["code"],
            int(item["scenario"][1:]),
            int(item["design"]["seed"]),
        )
    )
    return records


def build_dataset(runs: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    expected_pairs = {(regime["code"], scenario) for regime in REGIMES.values() for scenario in SCENARIOS}
    observed_pairs = {(run["regime"]["code"], run["scenario"]) for run in runs}
    missing_pairs = sorted(expected_pairs - observed_pairs)
    if missing_pairs:
        raise SystemExit(f"Missing completed R1/R2 scenario pairs: {missing_pairs}")

    frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    metric_headers: set[tuple[str, ...]] = set()
    for run in runs:
        summary = run["summary"]
        design = run["design"]
        scenario = run["scenario"]
        regime = run["regime"]
        metrics_path = run["metrics_path"]
        summary_path = run["summary_path"]
        frame = pd.read_csv(metrics_path)
        metric_headers.add(tuple(frame.columns))

        configured_months = int(design["requested_final_month"])
        expected_months = list(range(1, configured_months + 1))
        actual_months = frame["month"].astype(int).tolist()
        if actual_months != expected_months:
            raise SystemExit(
                f"{metrics_path} does not contain one ordered row for every month 1-{configured_months}"
            )
        if "scenario" in frame.columns:
            if set(frame["scenario"].astype(str)) != {scenario}:
                raise SystemExit(f"Scenario mismatch in {metrics_path}")
            frame = frame.drop(columns=["scenario"])
        if "population" in frame.columns:
            if set(frame["population"].astype(int)) != {int(design["population"])}:
                raise SystemExit(f"Population mismatch in {metrics_path}")
            frame = frame.drop(columns=["population"])

        decision = summary.get("decision_audit", {})
        behavior = decision.get("behavior_qualification", {})
        pre = summary.get("pre_equilibrium_audit", {})
        provider = summary.get("provider", {})
        model_provenance = decision.get("model_provenance", {})
        run_id = (
            f"{regime['code']}_{scenario}_N{int(design['population']):05d}_"
            f"M{configured_months:03d}_S{int(design['seed']):03d}"
        )
        months_since_activation = frame["month"].astype(int) - ACTIVATION_MONTH
        provenance: dict[str, Any] = {
            "dataset_schema_version": "1.0.0",
            "study_stage": "formal",
            "scenario_definition_version": summary["scenario_definition_version"],
            "regime_code": regime["code"],
            "regime_name": regime["name"],
            "regime_definition": regime["definition"],
            "scenario_code": scenario,
            "scenario_name": SCENARIOS[scenario][0],
            "scenario_definition": SCENARIOS[scenario][1],
            "run_id": run_id,
            "seed": int(design["seed"]),
            "population": int(design["population"]),
            "configured_months": configured_months,
            "activation_month": ACTIVATION_MONTH,
            "provider": provider.get("provider"),
            "requested_model": provider.get("model"),
            "api_base": provider.get("api_base"),
            "llm_roles_json": json_compact(summary.get("llm_roles", [])),
            "response_model_pairs_json": json_compact(model_provenance.get("pairs", {})),
            "source_fingerprint": summary.get("source_fingerprint"),
            "run_status": summary.get("status"),
            "run_relative_path": str(run["run_dir"].relative_to(ROOT)),
            "metrics_source_sha256": sha256(metrics_path),
            "summary_source_sha256": sha256(summary_path),
            "decision_records": decision.get("records"),
            "decision_accepted": decision.get("accepted"),
            "decision_fallbacks": decision.get("fallbacks"),
            "decision_fallback_rate": decision.get("fallback_rate"),
            "decision_llm_records": decision.get("llm_records"),
            "decision_expected_llm_records": decision.get("expected_llm_records"),
            "behavior_suitable_for_claims": behavior.get("suitable_for_behavior_claims"),
            "behavior_qualification_json": json_compact(behavior),
            "pre_equilibrium_accounting_pass": pre.get("accounting_pass"),
            "pre_equilibrium_boundary_pass": pre.get("boundary_pass"),
            "pre_equilibrium_warmup_stability_pass": pre.get("warmup_stability_pass"),
            "pre_equilibrium_path_gate_pass": pre.get("path_gate_pass"),
            "pre_equilibrium_max_accounting_absolute_error": pre.get(
                "max_accounting_absolute_error"
            ),
            "pre_equilibrium_max_accounting_tolerance_ratio": pre.get(
                "max_accounting_tolerance_ratio"
            ),
        }
        provenance_frame = pd.DataFrame(
            {column: [value] * len(frame) for column, value in provenance.items()}
        )
        provenance_frame["phase"] = frame["month"].astype(int).map(
            lambda month: "pre_equilibrium" if month < ACTIVATION_MONTH else "post_shock"
        )
        provenance_frame["months_since_activation"] = months_since_activation
        provenance_frame["policy_active"] = frame["month"].astype(int) >= ACTIVATION_MONTH
        frame = pd.concat(
            [provenance_frame.reset_index(drop=True), frame.reset_index(drop=True)],
            axis=1,
        )
        frames.append(frame)
        manifest_rows.append(
            {
                "run_id": run_id,
                "regime": regime["code"],
                "scenario": scenario,
                "seed": int(design["seed"]),
                "rows": len(frame),
                "columns": len(frame.columns),
                "status": summary.get("status"),
                "fallbacks": decision.get("fallbacks"),
                "source_fingerprint": summary.get("source_fingerprint"),
                "metrics_sha256": provenance["metrics_source_sha256"],
                "source_path": provenance["run_relative_path"],
            }
        )

    if len(metric_headers) != 1:
        raise SystemExit(f"Completed runs use {len(metric_headers)} distinct metrics.csv schemas")
    dataset = pd.concat(frames, ignore_index=True).copy()
    regime_order = {"R1": 1, "R2": 2}
    dataset["_regime_sort"] = dataset["regime_code"].map(regime_order)
    dataset["_scenario_sort"] = dataset["scenario_code"].str[1:].astype(int)
    dataset = (
        dataset.sort_values(["_regime_sort", "_scenario_sort", "seed", "month"])
        .drop(columns=["_regime_sort", "_scenario_sort"])
        .reset_index(drop=True)
    )
    key_columns = ["regime_code", "scenario_code", "seed", "month"]
    if dataset.duplicated(key_columns).any():
        raise SystemExit(f"Duplicate primary keys found: {key_columns}")
    return dataset, manifest_rows


def primary_metric_rows() -> list[tuple[str, str, str]]:
    return [
        (
            "居民福利",
            "bottom60_real_consumption",
            "对post_shock月份求和；政策比较使用相同seed和正确情景对照。",
        ),
        (
            "就业转型",
            "unemployment_rate; employment_rate",
            "失业峰值取post_shock最大值；尾期就业率取最后12个月均值。",
        ),
        (
            "家庭安全",
            "liquidity_vulnerable_rate",
            "尾期指标取最后12个月均值；比例在文件中以0-1编码。",
        ),
        (
            "社会回报",
            "public_service_index",
            "尾期公共服务取最后12个月均值。",
        ),
        (
            "产业生态",
            "firm_count; market_hhi",
            "企业数取最后一个月；HHI建议报告最后12个月均值。",
        ),
        (
            "财政可持续",
            "government_debt_ratio",
            "最高债务率取post_shock期间最大值。",
        ),
    ]


def write_document(
    dataset: pd.DataFrame,
    runs: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    generated_at: str,
) -> None:
    seeds = sorted(dataset["seed"].unique().tolist())
    lines: list[str] = [
        "# Institutional v2 R1/R2 E0-E6 Monthly Dataset",
        "",
        "## 1. Canonical files",
        "",
        f"- Data: `{DATA_PATH.relative_to(ROOT)}`",
        f"- Documentation: `{DOC_PATH.relative_to(ROOT)}`",
        f"- Generated at: `{generated_at}`",
        f"- Data SHA-256: `{sha256(DATA_PATH)}`",
        "",
        "这两个文件构成当前论文绘图和二次分析的规范数据接口。Parquet 文件为唯一数据表，本文档定义其范围、键、单位、情景语义和读取方法。",
        "",
        "## 2. Scope and grain",
        "",
        f"- Rows: **{len(dataset):,}**",
        f"- Columns: **{len(dataset.columns):,}**（含运行溯源字段与原始月度指标）",
        f"- Completed runs: **{len(runs)}**",
        f"- Regimes: **{', '.join(sorted(dataset['regime_code'].unique()))}**",
        f"- Scenarios: **{', '.join(sorted(dataset['scenario_code'].unique(), key=lambda value: int(value[1:])))}**",
        f"- Seeds found: **{', '.join(map(str, seeds))}**",
        f"- Months per run: **{', '.join(map(str, sorted(dataset['configured_months'].unique())))}**",
        "- Row grain: one `regime_code × scenario_code × seed × month` observation.",
        "- Primary key: `regime_code`, `scenario_code`, `seed`, `month`.",
        f"- Treatment timing: months 1-{ACTIVATION_MONTH - 1} are `pre_equilibrium`; month {ACTIVATION_MONTH} onward is `post_shock`.",
        "- Rates, ratios, and shares are stored as proportions in `[0,1]`; multiply by 100 only for percentage display.",
        "",
        "当前规范数据集只纳入研究设计中保留的 R1 和 R2。R0 是规则基准架构，R3 当前未形成完整 E0-E6 可比矩阵，因而不进入该文件。",
        "",
        "## 3. Cognitive regimes",
        "",
        "| Code | Name | Definition |",
        "|---|---|---|",
    ]
    for regime in REGIMES.values():
        lines.append(f"| {regime['code']} | {regime['name']} | {regime['definition']} |")
    lines += [
        "",
        "## 4. Institutional scenarios",
        "",
        "| Scenario | Name | Definition | Correct comparison |",
        "|---|---|---|---|",
    ]
    comparisons = {
        "E0": "time counterfactual",
        "E1": "E1 - E0",
        "E2": "E2 - E1",
        "E3": "E3 - E1",
        "E4": "E4 - E1",
        "E5": "E5 - E1 and mechanism decomposition",
        "E6": "E6 - E5",
    }
    for scenario, (name, definition) in SCENARIOS.items():
        lines.append(f"| {scenario} | {name} | {definition} | {comparisons[scenario]} |")
    lines += [
        "",
        "## 5. Three-layer evaluation framework",
        "",
        "### 5.1 Core outcomes",
        "",
        "| Dimension | Columns | Recommended aggregation |",
        "|---|---|---|",
    ]
    for dimension, columns, aggregation in primary_metric_rows():
        lines.append(f"| {dimension} | `{columns}` | {aggregation} |")
    lines += [
        "",
        "### 5.2 Mechanism indicators",
        "",
        "- **E2 employment responsibility:** `ai_attributable_layoffs_blocked`, `cumulative_ai_attributable_layoffs_blocked`, `average_required_work_hours`, `average_work_intensity`, `government_retention_wage_subsidy`, `distress_exemption_layoffs`, `firm_exit_layoffs`.",
        "- **E3 AI levy and social return:** `ai_infrastructure_levy`, `cumulative_ai_levy_revenue`, `government_ai_levy_public_service_spending`, `government_ai_levy_public_investment`, `government_ai_levy_fund_balance`, `public_service_index`.",
        "- **E4 time dividend and solo enterprise:** `self_employment_rate`, `solo_entries`, `solo_exits`, `solo_enterprise_sales`, `solo_enterprise_income`, `solo_b2b_sales`, `solo_external_sales`, `solo_net_additional_demand`, `solo_incumbent_displacement`.",
        "- **E5 integrated compact:** jointly read all E2-E4 fields plus `firm_count`, `firm_exits`, `market_hhi`, household resilience, and fiscal fields.",
        "- **E6 fiscal constraint:** compare with E5 using `government_debt_ratio`, `government_fiscal_curtailment`, `government_statutory_funding_gap`, `government_arrears`, and `public_service_index`.",
        "",
        "### 5.3 Validity and model audit",
        "",
        "Run-level audit fields are repeated on each monthly row so that filtered extracts retain their own provenance. Use `pre_equilibrium_*`, `decision_*`, `behavior_*`, `requested_model`, `response_model_pairs_json`, `source_fingerprint`, and the two source SHA-256 columns.",
        "",
        "The current file contains only seed 1. Multi-seed means, intervals, and sign consistency must be computed only after additional completed seed directories exist and the export is rerun.",
        "",
        "## 6. Reading and plotting",
        "",
        "### Python / Pandas",
        "",
        "```python",
        "import pandas as pd",
        "",
        f"df = pd.read_parquet('{DATA_PATH.relative_to(ROOT).as_posix()}')",
        "r2_e3 = df.query(\"regime_code == 'R2' and scenario_code == 'E3'\")",
        "r2_e3.plot(x='month', y=['unemployment_rate', 'employment_rate'])",
        "```",
        "",
        "### Correct paired scenario contrast",
        "",
        "```python",
        "post = df.query(\"regime_code == 'R2' and phase == 'post_shock'\")",
        "cum = post.groupby(['scenario_code', 'seed'])['bottom60_real_consumption'].sum()",
        "e3_vs_e1 = cum.loc['E3'] / cum.loc['E1'] - 1",
        "```",
        "",
        "### R1-R2 architecture comparison",
        "",
        "```python",
        "e5 = df.query(\"scenario_code == 'E5' and phase == 'post_shock'\")",
        "peak_u = e5.groupby(['regime_code', 'seed'])['unemployment_rate'].max()",
        "```",
        "",
        "## 7. Run manifest",
        "",
        "| Run | Rows | Status | Fallbacks | Source fingerprint | Metrics SHA-256 |",
        "|---|---:|---|---:|---|---|",
    ]
    for item in manifest_rows:
        lines.append(
            f"| {item['run_id']} | {item['rows']} | {item['status']} | "
            f"{item['fallbacks']} | `{item['source_fingerprint']}` | `{item['metrics_sha256']}` |"
        )
    lines += [
        "",
        "## 8. Complete column dictionary",
        "",
        "| Column | Group | Type | Unit / encoding | Description |",
        "|---|---|---|---|---|",
    ]
    for column in dataset.columns:
        dtype = dataset[column].dtype
        lines.append(
            f"| `{column}` | {classify_column(column)} | `{dtype}` | "
            f"{infer_unit(column, dtype)} | {describe_column(column)} |"
        )
    lines += [
        "",
        "## 9. Validation contract",
        "",
        "- Every retained run must have status `completed` in both `summary.json` and `run_manifest.json`.",
        "- R1 and R2 must each contain E0-E6.",
        "- Every run must contain exactly one ordered monthly row from month 1 through `configured_months`.",
        "- All runs must share the same raw monthly-metric schema.",
        "- The primary key must be unique.",
        "- The Parquet file must be readable and reproduce the documented row/column counts.",
        "- Source hashes in this document and the Parquet file allow later drift detection.",
        "",
    ]
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = discover_completed_runs()
    if not runs:
        raise SystemExit("No completed institutional_v2 R1/R2 runs were found")
    dataset, manifest_rows = build_dataset(runs)
    generated_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "dataset_schema_version": "1.0.0",
        "title": "Institutional v2 R1/R2 E0-E6 monthly economic metrics",
        "row_grain": "regime_code x scenario_code x seed x month",
        "primary_key": json_compact(["regime_code", "scenario_code", "seed", "month"]),
        "activation_month": str(ACTIVATION_MONTH),
        "regimes": json_compact({value["code"]: value for value in REGIMES.values()}),
        "scenarios": json_compact(SCENARIOS),
        "generated_at_utc": generated_at,
        "documentation": str(DOC_PATH.relative_to(ROOT)),
        "source_root": str(RESULTS.relative_to(ROOT)),
    }
    table = pa.Table.from_pandas(dataset, preserve_index=False)
    existing = table.schema.metadata or {}
    table = table.replace_schema_metadata(
        {**existing, **{key.encode(): value.encode() for key, value in metadata.items()}}
    )
    pq.write_table(table, DATA_PATH, compression="zstd", write_statistics=True)
    write_document(dataset, runs, manifest_rows, generated_at)

    reloaded = pd.read_parquet(DATA_PATH)
    if reloaded.shape != dataset.shape:
        raise SystemExit(
            f"Parquet round-trip mismatch: expected {dataset.shape}, got {reloaded.shape}"
        )
    print(f"Data: {DATA_PATH}")
    print(f"Documentation: {DOC_PATH}")
    print(f"Shape: {dataset.shape[0]} rows x {dataset.shape[1]} columns")
    print(f"Seeds: {sorted(dataset['seed'].unique().tolist())}")
    print(f"Data SHA-256: {sha256(DATA_PATH)}")


if __name__ == "__main__":
    main()
