from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Resident:
    id: int
    income_group: str
    base_productivity: float
    employed: bool
    employer_id: int | None
    gross_wage: float
    baseline_gross_wage: float
    other_baseline_income: float
    monthly_disposable_anchor: float
    baseline_consumption: float
    consumption_propensity: float
    minimum_consumption: float
    cash: float
    initial_cash: float
    deposits: float = 0.0
    managed_fund_assets: float = 0.0
    deposit_flow: float = 0.0
    managed_fund_flow: float = 0.0
    withdrawal_flow: float = 0.0
    baseline_disposable_income: float = 0.0
    shock_unemployed: bool = False
    # General unemployment spell length, regardless of why the resident is
    # unemployed. This feeds descriptive/persistent-unemployment metrics.
    unemployment_duration: int = 0
    # Duration of model-generated displacement only. This preserves the
    # calibrated distinction between baseline structural unemployment and an
    # income/consumption shock caused by layoffs or firm exit.
    shock_unemployment_duration: int = 0
    monthly_work_hours: float = 160.0
    redistributed_income: float = 0.0
    baseline_transfer: float = 0.0
    disposable_income: float = 0.0
    nominal_consumption: float = 0.0
    real_consumption: float = 0.0
    last_net_wage: float = 0.0
    bank_interest_income: float = 0.0
    consumption_stance: str = "normal"
    target_cash: float = 0.0
    personal_ai_use_rate: float = 0.0
    personal_ai_spending: float = 0.0
    personal_ai_productivity_gain: float = 0.0
    # Institutional experiment fields.  A self-employed resident remains
    # economically employed but is no longer attached to a wage-paying firm.
    self_employed: bool = False
    saved_ai_hours: float = 0.0
    entrepreneurial_readiness: float = 0.0
    solo_business_months: int = 0
    solo_loss_months: int = 0
    solo_business_revenue: float = 0.0
    solo_business_income: float = 0.0
    cumulative_solo_business_revenue: float = 0.0
    cumulative_solo_business_income: float = 0.0


@dataclass
class Firm:
    id: int
    firm_type: str
    size_tier: str
    employee_ids: list[int]
    ai_target: float
    # Deprecated checkpoint-compatibility field. AI-related labor savings are
    # represented exclusively by ai_multiplier in the production function.
    labor_displacement: float
    culture: str = "adaptive"
    ai_multiplier: float = 1.0
    expected_demand: float = 0.0
    capacity: float = 0.0
    sales: float = 0.0
    price: float = 1.0
    wage_bill: float = 0.0
    fixed_cost: float = 0.0
    cash: float = 0.0
    initial_cash: float = 0.0
    pre_tax_profit: float = 0.0
    retained_profit: float = 0.0
    market_share: float = 0.0
    baseline_quantity: float = 0.0
    baseline_price: float = 1.0
    bank_debt: float = 0.0
    investment_capital: float = 0.0
    planned_investment: float = 0.0
    actual_investment: float = 0.0
    principal_repayment: float = 0.0
    interest_payment: float = 0.0
    delinquency_months: int = 0
    loan_status: str = "normal"
    credit_default_probability: float = 0.0
    loan_writeoff: float = 0.0
    collateral_recovery: float = 0.0
    repayment_suspension_months: int = 0
    distressed_months: int = 0
    labor_stance: str = "baseline"
    founding_month: int = 0
    initial_employee_count: int = 0
    peak_employee_count: int = 0
    loss_months: int = 0
    # Employment-responsibility and AI-infrastructure-levy audit fields.
    ai_labor_demand: float = 0.0
    shadow_no_ai_labor_demand: float = 0.0
    protected_job_floor: int = 0
    ai_attributable_layoffs_blocked: int = 0
    work_intensity: float = 1.0
    retention_wage_subsidy: float = 0.0
    cumulative_retention_wage_subsidy: float = 0.0
    restructuring_months: int = 0
    distress_exemption_layoffs: int = 0
    ai_levy_per_unit: float = 0.0
    ai_levy_paid: float = 0.0
    ai_levy_rent_base: float = 0.0


@dataclass
class Government:
    id: int = 20001
    ai_use_rate: float = 0.30
    cash: float = 0.0
    initial_cash: float = 0.0
    debt: float = 0.0
    arrears: float = 0.0
    arrears_incurred: float = 0.0
    arrears_repayment: float = 0.0
    public_service_spending: float = 0.0
    procurement: float = 0.0
    ai_spending: float = 0.0
    transfers: float = 0.0
    tax_revenue: float = 0.0
    fiscal_balance: float = 0.0
    public_service_index: float = 1.0
    policy_stance: str = "baseline"
    fiscal_shortfall: float = 0.0
    fiscal_curtailment: float = 0.0
    statutory_funding_gap: float = 0.0
    public_investment: float = 0.0
    public_capital: float = 0.0
    employment_support_procurement: float = 0.0
    productivity_dividend_procurement: float = 0.0
    ai_levy_revenue: float = 0.0
    cumulative_ai_levy_revenue: float = 0.0
    ai_levy_fund_balance: float = 0.0
    ai_levy_public_service_spending: float = 0.0
    ai_levy_public_investment: float = 0.0
    cumulative_ai_levy_public_service_spending: float = 0.0
    cumulative_ai_levy_public_investment: float = 0.0
    ai_levy_bridge_advance: float = 0.0
    retention_wage_subsidy: float = 0.0
    cumulative_retention_wage_subsidy: float = 0.0

    @property
    def total_liabilities(self) -> float:
        return self.debt + self.arrears


@dataclass
class Bank:
    """Deterministic financial intermediary; monetary amounts remain core-owned."""

    id: int = 30001
    reserves: float = 0.0
    deposits: float = 0.0
    managed_funds: float = 0.0
    firm_loans: float = 0.0
    government_loans: float = 0.0
    other_financial_assets: float = 0.0
    equity: float = 0.0
    provisions: float = 0.0
    firm_credit_disbursed: float = 0.0
    firm_credit_requested: float = 0.0
    firm_credit_rejected: float = 0.0
    government_credit_disbursed: float = 0.0
    principal_repayments: float = 0.0
    other_asset_purchases: float = 0.0
    interest_income: float = 0.0
    interest_expense: float = 0.0
    provision_expense: float = 0.0
    writeoffs: float = 0.0
    recoveries: float = 0.0

    def balance_sheet_error(self) -> float:
        return (
            self.reserves
            + self.firm_loans
            + self.government_loans
            + self.other_financial_assets
            - self.provisions
            - self.deposits
            - self.managed_funds
            - self.equity
        )


@dataclass
class EconomyState:
    month: int
    scenario: str
    seed: int
    residents: dict[int, Resident]
    firms: dict[int, Firm]
    government: Government
    bank: Bank
    baseline_household_demand: float
    baseline_government_purchase: float
    baseline_total_output: float
    productivity_scale: float
    baseline_public_service: float = 0.0
    baseline_procurement: float = 0.0
    baseline_ai_spending: float = 0.0
    intents: dict[int, dict[str, Any]] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    initial_firm_count: int = 0
    next_firm_id: int = 10001
    cumulative_firm_entries: int = 0
    cumulative_firm_exits: int = 0
    cumulative_entry_jobs: int = 0
    cumulative_exit_jobs: int = 0
    price_index_basket: dict[int, dict[str, float]] = field(default_factory=dict)
    initial_culture_employment: dict[str, int] = field(default_factory=dict)
    cumulative_ai_attributable_layoffs_blocked: int = 0
    cumulative_distress_exemption_layoffs: int = 0
    cumulative_solo_entries: int = 0
    cumulative_solo_exits: int = 0
    cumulative_voluntary_wage_exits: int = 0
    cumulative_solo_substitution_sales: float = 0.0
    cumulative_solo_b2b_sales: float = 0.0
    cumulative_solo_induced_demand_sales: float = 0.0
    cumulative_solo_external_sales: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "month": self.month,
            "scenario": self.scenario,
            "seed": self.seed,
            "residents": {str(k): asdict(v) for k, v in self.residents.items()},
            "firms": {str(k): asdict(v) for k, v in self.firms.items()},
            "government": asdict(self.government),
            "bank": asdict(self.bank),
            "baseline_household_demand": self.baseline_household_demand,
            "baseline_government_purchase": self.baseline_government_purchase,
            "baseline_total_output": self.baseline_total_output,
            "productivity_scale": self.productivity_scale,
            "baseline_public_service": self.baseline_public_service,
            "baseline_procurement": self.baseline_procurement,
            "baseline_ai_spending": self.baseline_ai_spending,
            "intents": self.intents,
            "history": self.history,
            "initial_firm_count": self.initial_firm_count,
            "next_firm_id": self.next_firm_id,
            "cumulative_firm_entries": self.cumulative_firm_entries,
            "cumulative_firm_exits": self.cumulative_firm_exits,
            "cumulative_entry_jobs": self.cumulative_entry_jobs,
            "cumulative_exit_jobs": self.cumulative_exit_jobs,
            "price_index_basket": {
                str(k): dict(v) for k, v in self.price_index_basket.items()
            },
            "initial_culture_employment": dict(self.initial_culture_employment),
            "cumulative_ai_attributable_layoffs_blocked": (
                self.cumulative_ai_attributable_layoffs_blocked
            ),
            "cumulative_distress_exemption_layoffs": (
                self.cumulative_distress_exemption_layoffs
            ),
            "cumulative_solo_entries": self.cumulative_solo_entries,
            "cumulative_solo_exits": self.cumulative_solo_exits,
            "cumulative_voluntary_wage_exits": (
                self.cumulative_voluntary_wage_exits
            ),
            "cumulative_solo_substitution_sales": (
                self.cumulative_solo_substitution_sales
            ),
            "cumulative_solo_b2b_sales": self.cumulative_solo_b2b_sales,
            "cumulative_solo_induced_demand_sales": (
                self.cumulative_solo_induced_demand_sales
            ),
            "cumulative_solo_external_sales": (
                self.cumulative_solo_external_sales
            ),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EconomyState":
        return cls(
            month=int(raw["month"]),
            scenario=str(raw["scenario"]),
            seed=int(raw["seed"]),
            residents={
                int(k): Resident(
                    **{
                        **v,
                        "target_cash": float(
                            v.get("target_cash", v.get("initial_cash", 0.0))
                        ),
                    }
                )
                for k, v in raw["residents"].items()
            },
            firms={
                int(k): Firm(
                    **{
                        **v,
                        "culture": str(v.get("culture", "adaptive")),
                        "initial_employee_count": int(
                            v.get("initial_employee_count", len(v["employee_ids"]))
                        ),
                        "peak_employee_count": int(
                            v.get("peak_employee_count", len(v["employee_ids"]))
                        ),
                    }
                )
                for k, v in raw["firms"].items()
            },
            government=Government(**raw["government"]),
            bank=Bank(**raw.get("bank", {})),
            baseline_household_demand=float(raw["baseline_household_demand"]),
            baseline_government_purchase=float(raw["baseline_government_purchase"]),
            baseline_total_output=float(raw["baseline_total_output"]),
            productivity_scale=float(raw["productivity_scale"]),
            baseline_public_service=float(raw.get("baseline_public_service", 0.0)),
            baseline_procurement=float(raw.get("baseline_procurement", 0.0)),
            baseline_ai_spending=float(raw.get("baseline_ai_spending", 0.0)),
            intents={int(k): v for k, v in raw.get("intents", {}).items()},
            history=list(raw.get("history", [])),
            initial_firm_count=int(raw.get("initial_firm_count", len(raw["firms"]))),
            next_firm_id=int(
                raw.get(
                    "next_firm_id",
                    max((int(k) for k in raw["firms"]), default=10000) + 1,
                )
            ),
            cumulative_firm_entries=int(raw.get("cumulative_firm_entries", 0)),
            cumulative_firm_exits=int(raw.get("cumulative_firm_exits", 0)),
            cumulative_entry_jobs=int(raw.get("cumulative_entry_jobs", 0)),
            cumulative_exit_jobs=int(raw.get("cumulative_exit_jobs", 0)),
            price_index_basket={
                int(k): {
                    "quantity": float(v["quantity"]),
                    "base_price": float(v["base_price"]),
                    "last_price": float(v.get("last_price", v["base_price"])),
                }
                for k, v in raw.get("price_index_basket", {}).items()
            }
            or {
                int(k): {
                    "quantity": float(v.get("baseline_quantity", 0.0)),
                    "base_price": float(v.get("baseline_price", 1.0)),
                    "last_price": float(v.get("price", 1.0)),
                }
                for k, v in raw["firms"].items()
            },
            initial_culture_employment={
                str(k): int(v)
                for k, v in raw.get("initial_culture_employment", {}).items()
            }
            or {
                culture: sum(
                    len(firm["employee_ids"])
                    for firm in raw["firms"].values()
                    if str(firm.get("culture", "adaptive")) == culture
                )
                for culture in {
                    str(firm.get("culture", "adaptive"))
                    for firm in raw["firms"].values()
                }
            },
            cumulative_ai_attributable_layoffs_blocked=int(
                raw.get("cumulative_ai_attributable_layoffs_blocked", 0)
            ),
            cumulative_distress_exemption_layoffs=int(
                raw.get("cumulative_distress_exemption_layoffs", 0)
            ),
            cumulative_solo_entries=int(raw.get("cumulative_solo_entries", 0)),
            cumulative_solo_exits=int(raw.get("cumulative_solo_exits", 0)),
            cumulative_voluntary_wage_exits=int(
                raw.get("cumulative_voluntary_wage_exits", 0)
            ),
            cumulative_solo_substitution_sales=float(
                raw.get("cumulative_solo_substitution_sales", 0.0)
            ),
            cumulative_solo_b2b_sales=float(
                raw.get("cumulative_solo_b2b_sales", 0.0)
            ),
            cumulative_solo_induced_demand_sales=float(
                raw.get("cumulative_solo_induced_demand_sales", 0.0)
            ),
            cumulative_solo_external_sales=float(
                raw.get("cumulative_solo_external_sales", 0.0)
            ),
        )
