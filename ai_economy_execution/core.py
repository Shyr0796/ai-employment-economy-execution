from __future__ import annotations

import math
import random
from statistics import mean
from typing import Any

from .metrics import resident_distribution_metrics
from .models import EconomyState, Firm, Resident


RESIDENT_STANCES = {"normal", "cautious", "defensive"}
FIRM_STANCES = {"patient", "baseline", "aggressive"}
GOVERNMENT_STANCES = {"baseline", "stabilize", "balanced_support", "fiscal_guard"}


def bounded_intent(role: str, intent: dict[str, Any] | None) -> dict[str, str]:
    intent = intent or {}
    if role == "resident":
        value = str(intent.get("consumption_stance", "normal"))
        return {"consumption_stance": value if value in RESIDENT_STANCES else "normal"}
    if role == "firm":
        value = str(intent.get("labor_stance", "baseline"))
        return {"labor_stance": value if value in FIRM_STANCES else "baseline"}
    if role == "government":
        value = str(intent.get("policy_stance", "baseline"))
        return {"policy_stance": value if value in GOVERNMENT_STANCES else "baseline"}
    raise ValueError(f"Unknown economic role: {role}")


def fixed_basket_price_index(
    firms: list[Firm],
    reference_basket: dict[int, dict[str, float]] | None = None,
) -> float:
    """Return a base-period-quantity Laspeyres price index."""
    if reference_basket:
        current_prices = {firm.id: firm.price for firm in firms}
        for firm_id, item in reference_basket.items():
            if firm_id in current_prices:
                item["last_price"] = current_prices[firm_id]
        base_expenditure = sum(
            item["base_price"] * item["quantity"]
            for item in reference_basket.values()
        )
        current_cost_of_base_basket = sum(
            item.get("last_price", item["base_price"]) * item["quantity"]
            for item in reference_basket.values()
        )
        return current_cost_of_base_basket / max(base_expenditure, 1e-9)
    base_expenditure = sum(firm.baseline_price * firm.baseline_quantity for firm in firms)
    if base_expenditure <= 0.0:
        # Backward-compatible fallback for legacy serialized states that do
        # not contain baseline basket fields.
        current_quantity = sum(firm.sales / max(firm.price, 1e-9) for firm in firms)
        current_expenditure = sum(firm.sales for firm in firms)
        return current_expenditure / max(current_quantity, 1e-9)
    current_cost_of_base_basket = sum(firm.price * firm.baseline_quantity for firm in firms)
    return current_cost_of_base_basket / base_expenditure


class EconomyEngine:
    """Seed-reproducible monthly clearing engine; agents submit only bounded stances."""

    def __init__(self, state: EconomyState, config: dict[str, Any]):
        self.state = state
        self.config = config
        self._last_exit_reason_counts = {
            "cash_insolvent": 0,
            "operating_failure": 0,
            "deep_contraction": 0,
        }
        self._retention_subsidy_by_firm: dict[int, float] = {}

    def observe(self, agent_id: int) -> dict[str, Any]:
        macro = self.current_macro()
        trends = self._recent_macro_trends()
        if agent_id in self.state.residents:
            resident = self.state.residents[agent_id]
            cash_months = resident.cash / max(resident.baseline_consumption, 1.0)
            target_cash_months = resident.target_cash / max(
                resident.baseline_consumption, 1.0
            )
            return {
                "role": "resident",
                "month": self.state.month,
                "employed": resident.employed,
                "income_group": resident.income_group,
                "cash_months": cash_months,
                "target_cash_months": target_cash_months,
                "cash_gap_months": cash_months - target_cash_months,
                "unemployment_duration": resident.unemployment_duration,
                "shock_unemployed": resident.shock_unemployed,
                "shock_unemployment_duration": (
                    resident.shock_unemployment_duration
                ),
                "last_disposable_income": resident.disposable_income,
                "baseline_disposable_income": resident.baseline_disposable_income,
                "income_gap_ratio": (
                    resident.disposable_income
                    / max(resident.baseline_disposable_income, 1.0)
                    - 1.0
                ),
                "unemployment_rate": macro["unemployment_rate"],
                "shock_active": self._policy_is_active(self.state.month),
                "trend_available": trends["trend_available"],
                "trend_window_months": trends["trend_window_months"],
                "unemployment_change_3m": trends["unemployment_change_3m"],
                "aggregate_price_change_3m": trends[
                    "aggregate_price_change_3m"
                ],
                "real_consumption_change_3m": trends[
                    "real_consumption_change_3m"
                ],
            }
        if agent_id in self.state.firms:
            firm = self.state.firms[agent_id]
            target_utilization = float(self.config["firms"]["target_utilization"])
            expected_utilization = firm.expected_demand / max(
                firm.capacity * firm.price, 1.0
            )
            return {
                "role": "firm",
                "month": self.state.month,
                "firm_type": firm.firm_type,
                "culture": firm.culture,
                "employees": len(firm.employee_ids),
                "expected_demand": firm.expected_demand,
                "capacity_value": firm.capacity * firm.price,
                "expected_utilization": expected_utilization,
                "target_utilization": target_utilization,
                "utilization_gap": expected_utilization - target_utilization,
                "cash_ratio": firm.cash / max(firm.initial_cash, 1.0),
                "ai_multiplier": firm.ai_multiplier,
                "unemployment_rate": macro["unemployment_rate"],
                "shock_active": self._policy_is_active(self.state.month),
                "trend_available": trends["trend_available"],
                "trend_window_months": trends["trend_window_months"],
                "firm_sales_change_3m": trends["firm_sales_change_3m"],
                "capacity_utilization_change_3m": trends[
                    "capacity_utilization_change_3m"
                ],
                "market_hhi_change_3m": trends["market_hhi_change_3m"],
            }
        if agent_id == self.state.government.id:
            government = self.state.government
            return {
                "role": "government",
                "month": self.state.month,
                "unemployment_rate": macro["unemployment_rate"],
                "target_unemployment_rate": float(
                    self.config["government"]["target_unemployment_rate"]
                ),
                "unemployment_gap": macro["unemployment_rate"]
                - float(self.config["government"]["target_unemployment_rate"]),
                "demand_ratio": macro["household_consumption"] / self.state.baseline_household_demand,
                "cash_ratio": government.cash / max(government.initial_cash, 1.0),
                "debt_ratio": macro["government_debt_ratio"],
                "public_service_index": government.public_service_index,
                "shock_active": self._policy_is_active(self.state.month),
                "trend_available": trends["trend_available"],
                "trend_window_months": trends["trend_window_months"],
                "unemployment_change_3m": trends["unemployment_change_3m"],
                "household_consumption_change_3m": trends[
                    "household_consumption_change_3m"
                ],
                "debt_ratio_change_3m": trends["debt_ratio_change_3m"],
            }
        raise KeyError(f"Unknown agent id {agent_id}")

    def submit_intent(self, agent_id: int, intent: dict[str, Any]) -> dict[str, str]:
        role = self.observe(agent_id)["role"]
        validated = bounded_intent(role, intent)
        self.state.intents[agent_id] = validated
        return validated

    def current_macro(self) -> dict[str, float]:
        if self.state.history:
            return dict(self.state.history[-1])
        employed = sum(resident.employed for resident in self.state.residents.values())
        population = len(self.state.residents)
        annual_output = 12.0 * self.state.baseline_total_output
        return {
            "month": float(self.state.month),
            "employment_rate": employed / population,
            "unemployment_rate": 1.0 - employed / population,
            "household_consumption": self.state.baseline_household_demand,
            "government_debt_ratio": self.state.government.total_liabilities
            / max(annual_output, 1.0),
        }

    def _recent_macro_trends(self, window: int = 3) -> dict[str, float | int | bool]:
        """Summarize a fixed trailing window without adding mutable memory state."""
        defaults: dict[str, float | int | bool] = {
            "trend_available": False,
            "trend_window_months": window,
            "unemployment_change_3m": 0.0,
            "aggregate_price_change_3m": 0.0,
            "real_consumption_change_3m": 0.0,
            "firm_sales_change_3m": 0.0,
            "capacity_utilization_change_3m": 0.0,
            "market_hhi_change_3m": 0.0,
            "household_consumption_change_3m": 0.0,
            "debt_ratio_change_3m": 0.0,
        }
        if len(self.state.history) < window:
            return defaults

        start = self.state.history[-window]
        latest = self.state.history[-1]

        def absolute_change(key: str) -> float:
            return float(latest.get(key, 0.0)) - float(start.get(key, 0.0))

        def relative_change(key: str) -> float:
            reference = float(start.get(key, 0.0))
            if abs(reference) <= 1e-12:
                return 0.0
            return float(latest.get(key, reference)) / reference - 1.0

        return {
            **defaults,
            "trend_available": True,
            "unemployment_change_3m": absolute_change("unemployment_rate"),
            "aggregate_price_change_3m": relative_change("aggregate_price"),
            "real_consumption_change_3m": relative_change("real_consumption"),
            "firm_sales_change_3m": relative_change("firm_sales"),
            "capacity_utilization_change_3m": absolute_change(
                "capacity_utilization"
            ),
            "market_hhi_change_3m": absolute_change("market_hhi"),
            "household_consumption_change_3m": relative_change(
                "household_consumption"
            ),
            "debt_ratio_change_3m": absolute_change("government_debt_ratio"),
        }

    def _policy_is_active(self, month: int | None = None) -> bool:
        effective_month = self.state.month + 1 if month is None else month
        return effective_month >= int(self.config["simulation"]["shock_month"])

    def _government_parameter(self, key: str, month: int | None = None) -> float:
        if not self._policy_is_active(month) and key in self.config.get("pre_shock_government", {}):
            return float(self.config["pre_shock_government"][key])
        strategy = self._government_strategy(month)
        if self._policy_is_active(month) and key in strategy:
            return float(strategy[key])
        return float(self.config["government"][key])

    def _government_strategy(self, month: int | None = None) -> dict[str, Any]:
        government = self.config["government"]
        name = (
            str(government.get("policy_strategy", "passive_safety_net"))
            if self._policy_is_active(month)
            else "passive_safety_net"
        )
        strategies = government.get("policy_strategies", {})
        return dict(strategies.get(name, strategies.get("passive_safety_net", {})))

    def _culture_spec(self, firm: Firm) -> dict[str, Any]:
        cultures = self.config["firms"].get("cultures", [])
        for spec in cultures:
            if str(spec.get("name")) == firm.culture:
                return dict(spec)
        return {
            "name": firm.culture,
            "retention_commitment": 0.0,
            "layoff_demand_threshold": 0.0,
            "layoff_delay_months": 0,
            "ai_complementary_job_share": 0.0,
            "labor_adjustment_multiplier": 1.0,
            "price_aggressiveness": 1.0,
        }

    def _institutional_config(self, key: str) -> dict[str, Any]:
        return dict(
            self.config.get("institutional_experiments", {}).get(key, {})
        )

    def _institution_is_active(self, scenario_key: str) -> bool:
        return bool(
            self._policy_is_active()
            and self.config.get("scenario", {}).get(scenario_key, False)
        )

    def _levy_capture_rate(self, month: int | None = None) -> float:
        """Return the graduated AI-rent capture rate for the current month."""
        cfg = self._institutional_config("ai_infrastructure_levy")
        target = float(cfg.get("capture_rate", 0.0))
        initial = float(cfg.get("initial_capture_rate", target))
        ramp_months = max(int(cfg.get("capture_ramp_months", 0)), 0)
        if ramp_months == 0:
            return target
        active_month = (
            int(month if month is not None else self.state.month + 1)
            - int(self.config["simulation"]["shock_month"])
        )
        progress = min(max(active_month / ramp_months, 0.0), 1.0)
        return initial + (target - initial) * progress

    def _forecast_ai_levy(self, month: int) -> float:
        """Conservatively forecast current AI-rent revenue for same-month recycling."""
        if not self._institution_is_active("ai_infrastructure_levy"):
            return 0.0
        cfg = self._institutional_config("ai_infrastructure_levy")
        taxable_share = 1.0 - float(
            cfg.get("basic_consumption_exemption_share", 0.0)
        )
        capture_rate = (
            self._levy_capture_rate(month)
            if self._institution_is_active("ai_infrastructure_levy")
            else 0.0
        )
        forecast = 0.0
        for firm in self.state.firms.values():
            expected_sales = max(firm.expected_demand, firm.sales, 0.0)
            ai_rent_fraction = max(
                1.0 - 1.0 / max(firm.ai_multiplier, 1.0), 0.0
            )
            forecast += (
                expected_sales
                * ai_rent_fraction
                * taxable_share
                * capture_rate
            )
        return forecast

    def _plan_retention_wage_subsidies(self, month: int) -> dict[int, float]:
        """Share the wage cost of jobs retained specifically because of AI."""
        subsidies = {firm.id: 0.0 for firm in self.state.firms.values()}
        for firm in self.state.firms.values():
            firm.retention_wage_subsidy = 0.0
        if not self._institution_is_active("employment_responsibility"):
            return subsidies
        cfg = self._institutional_config("employment_responsibility")
        base_months = max(int(cfg.get("wage_cost_sharing_months", 0)), 0)
        extension_months = max(int(cfg.get("distress_extension_months", 0)), 0)
        elapsed = month - int(self.config["simulation"]["shock_month"])
        for firm in self.state.firms.values():
            warning = any(
                (
                    firm.loss_months
                    >= int(cfg.get("early_warning_loss_months", 4)),
                    firm.distressed_months
                    >= int(cfg.get("early_warning_cash_months", 4)),
                )
            )
            if not (
                elapsed < base_months
                or (warning and elapsed < base_months + extension_months)
            ):
                continue
            employees = [
                self.state.residents[resident_id]
                for resident_id in firm.employee_ids
            ]
            if not employees:
                continue
            average_effective_productivity = mean(
                [self._effective_productivity(item) for item in employees]
            )
            average_base_productivity = mean(
                [item.base_productivity for item in employees]
            )
            denominator = (
                float(self.config["firms"]["target_utilization"])
                * self.state.productivity_scale
                * firm.price
            )
            ai_required = firm.expected_demand / max(
                denominator
                * firm.ai_multiplier
                * average_effective_productivity,
                1.0,
            )
            shadow_required = firm.expected_demand / max(
                denominator * average_base_productivity,
                1.0,
            )
            protected_jobs = max(
                min(len(employees), math.ceil(shadow_required))
                - math.ceil(ai_required),
                0,
            )
            if protected_jobs <= 0:
                continue
            average_wage = firm.wage_bill / max(len(employees), 1)
            subsidies[firm.id] = min(
                protected_jobs
                * average_wage
                * float(cfg.get("wage_cost_sharing_rate", 0.0)),
                firm.wage_bill,
            )
        return subsidies

    @staticmethod
    def _effective_productivity(resident: Resident) -> float:
        return resident.base_productivity * (
            1.0
            + resident.personal_ai_productivity_gain
            * resident.personal_ai_use_rate
        )

    def _banking_enabled(self) -> bool:
        return bool(self.config.get("banking", {}).get("enabled", False))

    def _banking_active(self, month: int | None = None) -> bool:
        if not self._banking_enabled():
            return False
        effective_month = self.state.month + 1 if month is None else month
        activation_month = int(
            self.config["banking"].get(
                "activation_month", self.config["simulation"]["shock_month"]
            )
        )
        return effective_month >= activation_month

    def _bank_available_credit(self) -> float:
        if not self._banking_enabled():
            return math.inf
        bank = self.state.bank
        reserve_ratio = float(self.config["banking"].get("reserve_ratio", 0.10))
        managed_liquidity_ratio = float(
            self.config["banking"].get("managed_fund_liquidity_ratio", 0.05)
        )
        required_reserves = (
            reserve_ratio * bank.deposits
            + managed_liquidity_ratio * bank.managed_funds
        )
        return max(bank.reserves - required_reserves, 0.0)

    def _bank_risk_weighted_assets(self) -> float:
        bank = self.state.bank
        banking = self.config.get("banking", {})
        return (
            bank.firm_loans * float(banking.get("firm_loan_risk_weight", 1.0))
            + bank.government_loans
            * float(banking.get("government_bond_risk_weight", 0.20))
            + bank.other_financial_assets
            * float(banking.get("other_asset_risk_weight", 0.50))
        )

    def _bank_capital_adequacy_ratio(self) -> float:
        risk_weighted_assets = self._bank_risk_weighted_assets()
        if risk_weighted_assets <= 0.0:
            return 999.0 if self.state.bank.equity >= 0.0 else -999.0
        return self.state.bank.equity / risk_weighted_assets

    def _capital_credit_room(self, risk_weight: float) -> float:
        minimum_ratio = float(
            self.config["banking"].get("minimum_capital_adequacy_ratio", 0.12)
        )
        if minimum_ratio <= 0.0 or risk_weight <= 0.0:
            return math.inf
        maximum_risk_assets = max(self.state.bank.equity, 0.0) / minimum_ratio
        return max(
            (maximum_risk_assets - self._bank_risk_weighted_assets()) / risk_weight,
            0.0,
        )

    def _firm_default_probability(self, firm: Firm) -> float:
        banking = self.config["banking"]
        probability = float(
            banking.get("baseline_annual_default_probability", 0.015)
        )
        if firm.pre_tax_profit < 0.0 or firm.distressed_months > 0:
            probability *= float(banking.get("stress_default_multiplier", 3.0))
        probability *= 1.0 + 0.50 * firm.delinquency_months
        annual_sales = 12.0 * firm.baseline_quantity * firm.baseline_price
        probability *= 1.0 + firm.bank_debt / max(annual_sales, 1.0)
        return min(max(probability, 0.0), 1.0)

    def _start_bank_month(self) -> float:
        bank = self.state.bank
        bank.firm_credit_disbursed = 0.0
        bank.firm_credit_requested = 0.0
        bank.firm_credit_rejected = 0.0
        bank.government_credit_disbursed = 0.0
        bank.principal_repayments = 0.0
        bank.other_asset_purchases = 0.0
        bank.interest_income = 0.0
        bank.interest_expense = 0.0
        bank.provision_expense = 0.0
        bank.writeoffs = 0.0
        bank.recoveries = 0.0
        for resident in self.state.residents.values():
            resident.deposit_flow = 0.0
            resident.managed_fund_flow = 0.0
            resident.withdrawal_flow = 0.0
            resident.bank_interest_income = 0.0
        for firm in self.state.firms.values():
            firm.planned_investment = 0.0
            firm.actual_investment = 0.0
            firm.principal_repayment = 0.0
            firm.interest_payment = 0.0
            firm.loan_writeoff = 0.0
            firm.collateral_recovery = 0.0
        if not self._banking_enabled():
            return 0.0

        banking = self.config["banking"]
        depreciation = float(banking.get("capital_depreciation_rate", 0.0))
        for firm in self.state.firms.values():
            firm.investment_capital *= max(0.0, 1.0 - depreciation)
        public_depreciation = float(
            banking.get("public_capital_depreciation_rate", 0.0)
        )
        self.state.government.public_capital *= max(
            0.0, 1.0 - public_depreciation
        )

        if not self._banking_active():
            return 0.0

        available = self._bank_available_credit()
        funding = bank.deposits + bank.managed_funds
        maximum_firm_loans = funding * float(
            banking.get(
                "max_firm_loan_share",
                banking.get("target_firm_loan_share", 0.58),
            )
        )
        total_credit_room = max(
            funding * float(banking.get("maximum_loan_to_funding_ratio", 0.80))
            - bank.firm_loans
            - bank.government_loans,
            0.0,
        )
        firm_portfolio_room = max(maximum_firm_loans - bank.firm_loans, 0.0)
        adjustment_speed = float(banking.get("credit_adjustment_speed", 0.25))
        budget = min(
            available * adjustment_speed,
            firm_portfolio_room,
            total_credit_room,
            self._capital_credit_room(
                float(banking.get("firm_loan_risk_weight", 1.0))
            ),
        )
        threshold = float(banking.get("investment_utilization_threshold", 0.80))
        response = float(banking.get("investment_response", 0.75))
        replacement_rate = float(banking.get("replacement_investment_rate", 0.05))
        leverage_limit = float(banking.get("max_firm_debt_to_annual_sales", 0.50))
        concentration_limit = float(
            banking.get("max_single_borrower_share", 0.05)
        )
        minimum_dscr = float(
            banking.get("minimum_debt_service_coverage_ratio", 1.25)
        )
        annual_debt_service_rate = (
            12.0 * float(banking.get("firm_principal_repayment_rate", 0.008))
            + float(banking.get("loan_annual_interest_rate", 0.04))
        )
        maximum_pd = float(
            banking.get("maximum_approved_default_probability", 0.12)
        )
        desired: list[tuple[Firm, float]] = []
        for firm in self.state.firms.values():
            utilization = firm.sales / max(firm.capacity * firm.price, 1.0)
            annual_sales = 12.0 * firm.baseline_quantity * firm.baseline_price
            annual_cash_flow = max(firm.pre_tax_profit, 0.0) * 12.0
            cash_flow_debt_limit = annual_cash_flow / max(
                minimum_dscr * annual_debt_service_rate, 1e-9
            )
            approved_debt_limit = min(
                leverage_limit * annual_sales,
                concentration_limit * funding,
                cash_flow_debt_limit,
            )
            debt_room = max(
                approved_debt_limit - firm.bank_debt,
                0.0,
            )
            firm.credit_default_probability = self._firm_default_probability(firm)
            gross_request = (
                replacement_rate
                + max(utilization - threshold, 0.0) * response
            ) * firm.baseline_quantity * firm.baseline_price
            bank.firm_credit_requested += gross_request
            request = min(gross_request, debt_room)
            if request > 0.0 and firm.credit_default_probability <= maximum_pd:
                desired.append((firm, request))
        total_desired = sum(request for _, request in desired)
        scale = min(1.0, budget / total_desired) if total_desired > 0.0 else 0.0
        for firm, request in desired:
            loan = request * scale
            firm.bank_debt += loan
            firm.cash += loan
            firm.planned_investment = loan
            bank.reserves -= loan
            bank.firm_loans += loan
            bank.firm_credit_disbursed += loan
        bank.firm_credit_rejected = max(
            bank.firm_credit_requested - bank.firm_credit_disbursed, 0.0
        )
        return sum(firm.planned_investment for firm in self.state.firms.values())

    def _withdraw_deposit_for_consumption(self, resident: Resident, desired: float) -> None:
        if not self._banking_enabled() or resident.deposits <= 0.0:
            return
        cash_need = max(desired - resident.disposable_income - resident.cash, 0.0)
        withdrawal = min(cash_need, resident.deposits, self.state.bank.reserves)
        resident.deposits -= withdrawal
        resident.cash += withdrawal
        resident.withdrawal_flow += withdrawal
        self.state.bank.deposits -= withdrawal
        self.state.bank.reserves -= withdrawal

    def _resident_cash_target(self, resident: Resident) -> tuple[float, float]:
        household_cfg = self.config["households"]
        income_shortfall = max(
            1.0
            - resident.disposable_income
            / max(resident.baseline_disposable_income, 1.0),
            0.0,
        )
        stress_months = max(
            int(household_cfg.get("unemployment_stress_months", 6)), 1
        )
        unemployment_risk = (
            min(resident.shock_unemployment_duration / stress_months, 1.0)
            if resident.shock_unemployed and not resident.employed
            else 0.0
        )
        target = resident.initial_cash * (
            1.0
            + float(
                household_cfg.get("cash_target_income_shortfall_response", 1.0)
            )
            * income_shortfall
            + float(
                household_cfg.get("cash_target_unemployment_response", 1.0)
            )
            * unemployment_risk
        )
        return max(target, 0.0), unemployment_risk

    def _sweep_household_deposits(self) -> tuple[float, float, float]:
        if not self._banking_enabled():
            return 0.0, 0.0, 0.0
        bank = self.state.bank
        banking = self.config["banking"]
        household_cfg = self.config["households"]
        sweep_share = float(banking.get("deposit_sweep_share", 0.50))
        deposit_share = float(
            banking.get("household_deposit_allocation_share", 1.0)
        )
        managed_share = float(banking.get("household_managed_fund_share", 0.0))
        allocation_total = deposit_share + managed_share
        if allocation_total <= 0.0:
            deposit_share, managed_share = 1.0, 0.0
        else:
            deposit_share /= allocation_total
            managed_share /= allocation_total
        total_deposits = 0.0
        total_managed_funds = 0.0
        total_withdrawals = 0.0
        for resident in self.state.residents.values():
            target_cash, unemployment_risk = self._resident_cash_target(resident)
            resident.target_cash = target_cash
            if resident.cash < target_cash and resident.deposits > 0.0:
                rebalance = min(
                    (target_cash - resident.cash)
                    * float(household_cfg.get("cash_rebalance_speed", 0.50)),
                    resident.deposits,
                    bank.reserves,
                )
                resident.deposits -= rebalance
                resident.cash += rebalance
                resident.withdrawal_flow += rebalance
                bank.deposits -= rebalance
                bank.reserves -= rebalance
            if resident.cash > target_cash:
                risk_adjusted_sweep = sweep_share * max(
                    1.0
                    - float(
                        household_cfg.get("unemployment_sweep_reduction", 0.75)
                    )
                    * unemployment_risk,
                    0.0,
                )
                allocated = (resident.cash - target_cash) * risk_adjusted_sweep
                deposit = allocated * deposit_share
                managed_fund = allocated * managed_share
                resident.cash -= allocated
                resident.deposits += deposit
                resident.deposit_flow += deposit
                resident.managed_fund_assets += managed_fund
                resident.managed_fund_flow += managed_fund
                bank.deposits += deposit
                bank.managed_funds += managed_fund
                bank.reserves += allocated
            total_deposits += resident.deposit_flow
            total_managed_funds += resident.managed_fund_flow
            total_withdrawals += resident.withdrawal_flow
        return total_deposits, total_managed_funds, total_withdrawals

    def _settle_firm_investment(self, fulfillment: float) -> float:
        total = 0.0
        for firm in self.state.firms.values():
            investment = firm.planned_investment * fulfillment
            firm.actual_investment = investment
            firm.cash -= investment
            firm.investment_capital += investment
            total += investment
        return total

    def _service_firm_loans(self) -> float:
        if not self._banking_enabled():
            return 0.0
        banking = self.config["banking"]
        principal_rate = float(banking.get("firm_principal_repayment_rate", 0.01))
        monthly_interest_rate = float(
            banking.get("loan_annual_interest_rate", 0.04)
        ) / 12.0
        npl_months = int(banking.get("delinquency_months_to_npl", 3))
        bank = self.state.bank
        total = 0.0
        for firm in self.state.firms.values():
            if firm.bank_debt <= 0.0:
                firm.delinquency_months = 0
                firm.loan_status = "normal"
                continue
            scheduled_interest = firm.bank_debt * monthly_interest_rate
            scheduled_principal = firm.bank_debt * principal_rate
            if (
                bool(banking.get("enable_seeded_credit_events", True))
                and firm.repayment_suspension_months <= 0
            ):
                credit_rng = random.Random(
                    f"{self.state.seed}:{self.state.month + 1}:{firm.id}:credit"
                )
                if credit_rng.random() < firm.credit_default_probability / 12.0:
                    severe = credit_rng.random() < float(
                        banking.get("loss_given_default", 0.45)
                    )
                    firm.repayment_suspension_months = (
                        int(banking.get("writeoff_months", 12))
                        if severe
                        else npl_months
                    )
            if firm.repayment_suspension_months > 0:
                interest = 0.0
                repayment = 0.0
                firm.repayment_suspension_months -= 1
            else:
                available_cash = max(firm.cash, 0.0)
                interest = min(scheduled_interest, available_cash)
                available_cash -= interest
                repayment = min(scheduled_principal, available_cash)
            firm.cash -= interest + repayment
            firm.interest_payment = interest
            firm.bank_debt -= repayment
            firm.principal_repayment = repayment
            bank.firm_loans -= repayment
            bank.reserves += interest + repayment
            bank.interest_income += interest
            bank.equity += interest
            if interest + repayment + 1e-9 < scheduled_interest + scheduled_principal:
                firm.delinquency_months += 1
            else:
                firm.delinquency_months = 0
            if firm.delinquency_months == 0:
                firm.loan_status = "normal"
            elif firm.delinquency_months < npl_months:
                firm.loan_status = "watch"
            elif firm.delinquency_months < 6:
                firm.loan_status = "substandard"
            elif firm.delinquency_months < int(banking.get("writeoff_months", 12)):
                firm.loan_status = "doubtful"
            else:
                firm.loan_status = "loss"
            firm.credit_default_probability = self._firm_default_probability(firm)
            total += repayment
        bank.principal_repayments += total
        self._update_loan_provisions()
        self._write_off_bad_loans()
        self._update_loan_provisions()
        return total

    def _update_loan_provisions(self) -> float:
        bank = self.state.bank
        rates = self.config["banking"].get("provision_rates", {})
        target = sum(
            firm.bank_debt * float(rates.get(firm.loan_status, 0.01))
            for firm in self.state.firms.values()
        )
        change = target - bank.provisions
        bank.provisions = target
        bank.equity -= change
        bank.provision_expense += change
        return change

    def _write_off_bad_loans(self) -> float:
        bank = self.state.bank
        banking = self.config["banking"]
        writeoff_months = int(banking.get("writeoff_months", 12))
        loss_given_default = float(banking.get("loss_given_default", 0.45))
        total = 0.0
        for firm in self.state.firms.values():
            if firm.bank_debt <= 0.0 or firm.delinquency_months < writeoff_months:
                continue
            exposure = firm.bank_debt
            recovery = exposure * max(0.0, 1.0 - loss_given_default)
            provision_used = min(bank.provisions, exposure)
            bank.firm_loans -= exposure
            bank.provisions -= provision_used
            bank.reserves += recovery
            bank.equity += recovery - (exposure - provision_used)
            bank.writeoffs += exposure
            bank.recoveries += recovery
            firm.loan_writeoff = exposure
            firm.collateral_recovery = recovery
            firm.bank_debt = 0.0
            firm.delinquency_months = 0
            firm.repayment_suspension_months = 0
            firm.loan_status = "normal"
            total += exposure
        return total

    def _accrue_bank_returns_and_funding_costs(self) -> None:
        if not self._banking_active():
            return
        bank = self.state.bank
        banking = self.config["banking"]
        government_interest = bank.government_loans * float(
            banking.get("government_bond_annual_yield", 0.025)
        ) / 12.0
        liquid_asset_income = max(bank.reserves, 0.0) * float(
            banking.get("liquid_asset_annual_yield", 0.018)
        ) / 12.0
        other_income = bank.other_financial_assets * float(
            banking.get("other_asset_annual_yield", 0.03)
        ) / 12.0
        if government_interest > 0.0:
            bank.government_loans += government_interest
            self.state.government.debt += government_interest
            bank.equity += government_interest
            bank.interest_income += government_interest
        if other_income > 0.0:
            bank.reserves += other_income
            bank.equity += other_income
            bank.interest_income += other_income
        if liquid_asset_income > 0.0:
            bank.reserves += liquid_asset_income
            bank.equity += liquid_asset_income
            bank.interest_income += liquid_asset_income

        deposit_rate = float(banking.get("deposit_annual_interest_rate", 0.015)) / 12.0
        managed_rate = float(banking.get("managed_fund_annual_return", 0.025)) / 12.0
        funding_cost = 0.0
        for resident in self.state.residents.values():
            income = resident.deposits * deposit_rate + resident.managed_fund_assets * managed_rate
            resident.bank_interest_income = income
            funding_cost += income
        bank.reserves -= funding_cost
        bank.equity -= funding_cost
        bank.interest_expense += funding_cost

    def _rebalance_bank_residual_assets(self) -> float:
        """Sell residual assets only when the liquidity floor would otherwise be breached."""
        if not self._banking_enabled():
            return 0.0
        bank = self.state.bank
        funding = bank.deposits + bank.managed_funds
        banking = self.config["banking"]
        required_reserves = (
            float(banking.get("reserve_ratio", 0.10)) * bank.deposits
            + float(banking.get("managed_fund_liquidity_ratio", 0.05))
            * bank.managed_funds
        )
        target_reserves = max(
            required_reserves,
            float(
                banking.get(
                    "minimum_liquid_asset_share",
                    banking.get("target_liquid_asset_share", 0.15),
                )
            )
            * funding,
        )
        if bank.reserves >= target_reserves:
            return 0.0
        sale = min(target_reserves - bank.reserves, bank.other_financial_assets)
        bank.other_financial_assets -= sale
        bank.reserves += sale
        bank.other_asset_purchases = -sale
        return -sale

    def step(self) -> dict[str, Any]:
        state = self.state
        cfg = self.config
        household_cfg = cfg["households"]
        firm_cfg = cfg["firms"]
        gov_cfg = cfg["government"]
        scenario = cfg["scenario"]
        month = state.month + 1

        self._apply_intents()
        self._update_ai(month)
        (
            solo_entries,
            solo_exits,
            voluntary_wage_exits,
        ) = self._process_solo_enterprise_transitions(month)
        planned_firm_investment = self._start_bank_month()
        self._accrue_bank_returns_and_funding_costs()
        labor_tax_rate = float(household_cfg["labor_tax_rate"])

        gross_wages = 0.0
        for firm in state.firms.values():
            firm.wage_bill = 0.0
            wage_factor = 1.0 + float(firm_cfg["wage_productivity_pass_through"]) * (firm.ai_multiplier - 1.0)
            for resident_id in firm.employee_ids:
                resident = state.residents[resident_id]
                resident.gross_wage = resident.baseline_gross_wage * wage_factor
                resident.last_net_wage = resident.gross_wage * (1.0 - labor_tax_rate)
                firm.wage_bill += resident.gross_wage
            gross_wages += firm.wage_bill

        self._retention_subsidy_by_firm = (
            self._plan_retention_wage_subsidies(month)
        )
        lag = int(gov_cfg["policy_lag_months"])
        lagged = state.history[-lag] if len(state.history) >= lag else self.current_macro()
        government_plan = self._government_plan(lagged)
        government_plan = self._apply_fiscal_limit(government_plan)

        transfer_total = 0.0
        planned_consumption_total = 0.0
        cash_before = sum(resident.cash for resident in state.residents.values())
        for resident in state.residents.values():
            transfer = self._resident_transfer(resident, government_plan["transfer_extra_pool"])
            resident.redistributed_income = transfer
            transfer_total += transfer
            net_wage = resident.gross_wage * (1.0 - labor_tax_rate) if resident.employed else 0.0
            other_income = resident.other_baseline_income + resident.bank_interest_income
            if resident.shock_unemployed and not resident.employed:
                other_income *= float(
                    household_cfg.get("unemployment_other_income_retention", 1.0)
                )
            resident.disposable_income = net_wage + other_income + transfer
            income_gap = resident.disposable_income - resident.baseline_disposable_income
            desired = resident.baseline_consumption + resident.consumption_propensity * income_gap
            if resident.shock_unemployed:
                desired *= 1.0 - float(household_cfg["unemployed_consumption_penalty"])
                desired -= (
                    float(household_cfg["precautionary_response"])
                    * resident.baseline_consumption
                    * min(resident.shock_unemployment_duration / 6.0, 1.0)
                )
            stance_factor = {"normal": 1.0, "cautious": 0.94, "defensive": 0.86}[resident.consumption_stance]
            desired *= stance_factor
            desired = max(resident.minimum_consumption, desired)
            liquid_assets = resident.cash + resident.deposits
            personal_ai_cfg = household_cfg.get("personal_ai", {})
            desired_personal_ai = (
                float(personal_ai_cfg.get("monthly_cost", 0.0))
                * resident.personal_ai_use_rate
                if resident.employed and scenario.get("private_ai", False)
                else 0.0
            )
            total_desired = min(
                desired + desired_personal_ai,
                resident.disposable_income
                + float(household_cfg["cash_drawdown_rate"]) * liquid_assets,
            )
            self._withdraw_deposit_for_consumption(resident, total_desired)
            liquid_resources = resident.cash + resident.disposable_income
            resident.nominal_consumption = max(
                0.0, min(total_desired, liquid_resources)
            )
            resident.personal_ai_spending = min(
                desired_personal_ai, resident.nominal_consumption
            )
            planned_consumption_total += resident.nominal_consumption

        planned_government_purchase = (
            government_plan["public_service"]
            + government_plan["procurement"]
            + government_plan["ai_spending"]
            + government_plan["public_investment"]
        )
        solo_demand = self._plan_solo_demand(
            planned_consumption_total,
            planned_government_purchase,
            planned_firm_investment,
        )
        planned_consumption_total += solo_demand["induced"]
        total_orders = (
            planned_consumption_total
            + planned_government_purchase
            + planned_firm_investment
            + solo_demand["external"]
        )
        (
            solo_sales_channels,
            solo_income_tax,
            solo_enterprise_income,
        ) = self._allocate_solo_enterprise_sales(solo_demand)
        solo_enterprise_sales = sum(solo_sales_channels.values())
        firm_order_pool = max(total_orders - solo_enterprise_sales, 0.0)
        actual_sales = solo_enterprise_sales
        market_shares = self._normalized_market_shares()
        firms_for_orders = list(state.firms.values())
        targeted_support = min(
            government_plan.get("employment_support_procurement", 0.0),
            firm_order_pool,
        )
        eligible = [
            firm
            for firm in firms_for_orders
            if len(firm.employee_ids)
            >= max(1, math.ceil(0.85 * max(firm.initial_employee_count, 1)))
        ]
        if not eligible:
            targeted_support = 0.0
        eligible_share_total = sum(
            share
            for firm, share in zip(firms_for_orders, market_shares)
            if firm in eligible
        )
        general_orders = firm_order_pool - targeted_support
        for firm, share in zip(firms_for_orders, market_shares):
            support_share = (
                share / eligible_share_total
                if firm in eligible and eligible_share_total > 0.0
                else 0.0
            )
            order = general_orders * share + targeted_support * support_share
            firm.sales = min(order, firm.capacity * firm.price)
            actual_sales += firm.sales
        fulfillment = min(1.0, actual_sales / total_orders) if total_orders else 1.0

        aggregate_price = fixed_basket_price_index(
            list(state.firms.values()), state.price_index_basket
        )
        household_consumption = 0.0
        real_consumption = 0.0
        for resident in state.residents.values():
            resident.nominal_consumption *= fulfillment
            resident.personal_ai_spending *= fulfillment
            resident.real_consumption = (
                resident.nominal_consumption - resident.personal_ai_spending
            ) / max(aggregate_price, 1e-9)
            resident.cash += resident.disposable_income - resident.nominal_consumption
            household_consumption += resident.nominal_consumption
            real_consumption += resident.real_consumption
        (
            household_deposit_flow,
            household_managed_fund_flow,
            household_withdrawal_flow,
        ) = (
            self._sweep_household_deposits()
        )

        actual_public_service = government_plan["public_service"] * fulfillment
        actual_procurement = government_plan["procurement"] * fulfillment
        actual_ai_spending = government_plan["ai_spending"] * fulfillment
        actual_public_investment = government_plan["public_investment"] * fulfillment
        actual_government_purchase = (
            actual_public_service
            + actual_procurement
            + actual_ai_spending
            + actual_public_investment
        )
        actual_firm_investment = self._settle_firm_investment(fulfillment)

        indirect_tax_total = 0.0
        profit_tax_total = 0.0
        ai_levy_total = 0.0
        retained_profit_total = 0.0
        desired_retention_subsidy = sum(
            self._retention_subsidy_by_firm.values()
        )
        approved_retention_subsidy = float(
            government_plan.get("retention_wage_subsidy", 0.0)
        )
        retention_subsidy_scale = min(
            approved_retention_subsidy
            / max(desired_retention_subsidy, 1e-9),
            1.0,
        )
        capture_rate = (
            self._levy_capture_rate(month)
            if self._institution_is_active("ai_infrastructure_levy")
            else 0.0
        )
        levy_cfg = self._institutional_config("ai_infrastructure_levy")
        taxable_rent_share = 1.0 - float(
            levy_cfg.get("basic_consumption_exemption_share", 0.0)
        )
        for firm in state.firms.values():
            indirect_tax = firm.sales * float(gov_cfg["indirect_tax_rate"])
            firm.retention_wage_subsidy = (
                self._retention_subsidy_by_firm.get(firm.id, 0.0)
                * retention_subsidy_scale
            )
            firm.cumulative_retention_wage_subsidy += (
                firm.retention_wage_subsidy
            )
            operating_surplus_before_levy = (
                firm.sales
                - firm.wage_bill
                - firm.fixed_cost
                - indirect_tax
            )
            ai_rent_fraction = max(
                1.0 - 1.0 / max(firm.ai_multiplier, 1.0), 0.0
            )
            firm.ai_levy_rent_base = (
                min(
                    firm.sales * ai_rent_fraction * taxable_rent_share,
                    max(operating_surplus_before_levy, 0.0),
                )
                if self._institution_is_active("ai_infrastructure_levy")
                else 0.0
            )
            firm.ai_levy_paid = capture_rate * firm.ai_levy_rent_base
            firm.pre_tax_profit = (
                firm.sales
                - firm.wage_bill
                - firm.fixed_cost
                - indirect_tax
                - firm.ai_levy_paid
                + firm.retention_wage_subsidy
            )
            profit_tax = max(firm.pre_tax_profit, 0.0) * float(gov_cfg["firm_profit_tax_rate"])
            firm.retained_profit = firm.pre_tax_profit - profit_tax
            firm.cash += firm.retained_profit
            indirect_tax_total += indirect_tax
            profit_tax_total += profit_tax
            ai_levy_total += firm.ai_levy_paid
            retained_profit_total += firm.retained_profit
            firm.distressed_months = firm.distressed_months + 1 if firm.cash < 0 else 0
            firm.loss_months = firm.loss_months + 1 if firm.pre_tax_profit < 0 else 0
        firm_principal_repayment = self._service_firm_loans()
        settled_wage_bill = sum(firm.wage_bill for firm in state.firms.values())

        labor_tax = gross_wages * labor_tax_rate
        tax_revenue = (
            labor_tax
            + indirect_tax_total
            + profit_tax_total
            + solo_income_tax
            + ai_levy_total
        )
        retention_wage_subsidy_total = sum(
            firm.retention_wage_subsidy for firm in state.firms.values()
        )
        government_spending = (
            transfer_total
            + actual_government_purchase
            + retention_wage_subsidy_total
        )
        self._settle_government(tax_revenue, government_spending)
        government = state.government
        government.public_service_spending = actual_public_service
        government.procurement = actual_procurement
        government.employment_support_procurement = (
            government_plan.get("employment_support_procurement", 0.0) * fulfillment
        )
        government.productivity_dividend_procurement = (
            government_plan.get("productivity_dividend_procurement", 0.0)
            * fulfillment
        )
        government.ai_spending = actual_ai_spending
        government.public_investment = actual_public_investment
        government.public_capital += actual_public_investment
        actual_levy_public_service = (
            government_plan.get("ai_levy_public_service", 0.0) * fulfillment
        )
        actual_levy_public_investment = (
            government_plan.get("ai_levy_public_investment", 0.0) * fulfillment
        )
        government.ai_levy_revenue = ai_levy_total
        government.ai_levy_public_service_spending = actual_levy_public_service
        government.ai_levy_public_investment = actual_levy_public_investment
        government.retention_wage_subsidy = retention_wage_subsidy_total
        government.cumulative_retention_wage_subsidy += (
            retention_wage_subsidy_total
        )
        government.cumulative_ai_levy_revenue += ai_levy_total
        government.cumulative_ai_levy_public_service_spending += (
            actual_levy_public_service
        )
        government.cumulative_ai_levy_public_investment += (
            actual_levy_public_investment
        )
        government.ai_levy_fund_balance = (
            government.ai_levy_fund_balance
            + ai_levy_total
            - actual_levy_public_service
            - actual_levy_public_investment
        )
        government.ai_levy_bridge_advance = max(
            -government.ai_levy_fund_balance, 0.0
        )
        government.transfers = transfer_total
        government.tax_revenue = tax_revenue
        government.fiscal_balance = tax_revenue - government_spending
        self._rebalance_bank_residual_assets()

        ai_lag = int(gov_cfg["service_lag_months"])
        if len(state.history) >= ai_lag:
            lagged_ai = state.history[-ai_lag].get("government_ai_use_rate", float(gov_cfg["ai_use_rate"]))
        else:
            lagged_ai = float(gov_cfg["ai_use_rate"])
        baseline_service = state.baseline_public_service or (
            state.baseline_household_demand * float(gov_cfg["public_service_share"])
        )
        baseline_quality = 1.0 + float(gov_cfg["service_productivity"]) * float(gov_cfg["ai_use_rate"])
        public_capital_bonus = 1.0 + float(
            cfg.get("banking", {}).get(
                "public_capital_productivity_elasticity", 0.0
            )
        ) * government.public_capital / max(12.0 * baseline_service, 1.0)
        government.public_service_index = (
            actual_public_service / max(aggregate_price, 1e-9) * (1.0 + float(gov_cfg["service_productivity"]) * lagged_ai)
        ) / max(baseline_service * baseline_quality, 1e-9) * public_capital_bonus

        hired, fired = self._adjust_employment()
        routine_layoffs = fired
        for firm in state.firms.values():
            firm.peak_employee_count = max(
                firm.peak_employee_count,
                len(firm.employee_ids),
            )
        firm_exits, exit_jobs = self._process_firm_exits(month)
        fired += exit_jobs
        unmet_demand_ratio = max(total_orders - actual_sales, 0.0) / max(
            total_orders, 1.0
        )
        firm_entries, entry_jobs = self._process_firm_entries(
            month,
            unmet_demand_ratio,
            immediate=firm_exits > 0,
        )
        hired += entry_jobs
        self._refresh_capacities_and_shares()
        for resident in state.residents.values():
            if resident.employed:
                resident.unemployment_duration = 0
                resident.shock_unemployment_duration = 0
            else:
                resident.unemployment_duration += 1
                if resident.shock_unemployed:
                    resident.shock_unemployment_duration += 1
                else:
                    resident.shock_unemployment_duration = 0
                resident.gross_wage = 0.0
        employed_count = sum(
            resident.employed for resident in state.residents.values()
        )
        self_employed_count = sum(
            resident.self_employed for resident in state.residents.values()
        )
        wage_employed_count = sum(
            resident.employed and not resident.self_employed
            for resident in state.residents.values()
        )
        long_unemployed = sum(
            (not resident.employed) and resident.unemployment_duration >= 12
            for resident in state.residents.values()
        )

        annual_output = 12.0 * state.baseline_total_output
        annual_household_income = 12.0 * sum(
            resident.baseline_disposable_income
            for resident in state.residents.values()
        )
        bank_funding = state.bank.deposits + state.bank.managed_funds
        household_financial_wealth = sum(
            resident.cash + resident.deposits + resident.managed_fund_assets
            for resident in state.residents.values()
        )
        household_saving_flow = (
            sum(resident.disposable_income for resident in state.residents.values())
            - household_consumption
        )
        rolling_rows = state.history[-11:]
        rolling_12_household_saving = household_saving_flow + sum(
            float(row.get("household_saving_flow", 0.0)) for row in rolling_rows
        )
        rolling_12_nominal_output = actual_sales + sum(
            float(row.get("firm_sales", 0.0)) for row in rolling_rows
        )
        rolling_12_disposable_income = sum(
            resident.disposable_income for resident in state.residents.values()
        ) + sum(float(row.get("disposable_income", 0.0)) for row in rolling_rows)
        household_income_share = float(
            cfg.get("macro_calibration", {}).get(
                "household_disposable_income_share_of_gdp", 0.43
            )
        )
        rolling_12_macro_gdp = max(
            rolling_12_nominal_output,
            rolling_12_disposable_income / max(household_income_share, 1e-9),
        )
        if len(state.history) >= 12:
            comparison_row = state.history[-12]
            prior_financial_wealth = float(
                comparison_row.get("household_financial_wealth", 0.0)
            )
            prior_firm_loans = float(comparison_row.get("bank_firm_loans", 0.0))
            prior_government_bonds = float(
                comparison_row.get(
                    "bank_government_bonds",
                    comparison_row.get("bank_government_loans", 0.0),
                )
            )
        else:
            prior_financial_wealth = sum(
                resident.initial_cash for resident in state.residents.values()
            )
            prior_firm_loans = 0.0
            prior_government_bonds = 0.0
        rolling_12_financial_asset_accumulation = (
            household_financial_wealth - prior_financial_wealth
        )
        rolling_12_net_firm_credit = state.bank.firm_loans - prior_firm_loans
        rolling_12_net_government_bonds = (
            state.bank.government_loans - prior_government_bonds
        )
        nonperforming_loans = sum(
            firm.bank_debt
            for firm in state.firms.values()
            if firm.loan_status in {"substandard", "doubtful", "loss"}
        )
        bank_capital_ratio = self._bank_capital_adequacy_ratio()
        distribution_metrics = resident_distribution_metrics(
            list(state.residents.values()),
            float(household_cfg.get("liquidity_vulnerability_months", 3.0)),
            float(household_cfg.get("cash_vulnerability_months", 1.0)),
            float(household_cfg.get("consumption_stress_ratio", 0.85)),
            int(household_cfg.get("unemployment_stress_months", 6)),
        )
        firms = list(state.firms.values())
        normalized_shares = self._normalized_market_shares()
        market_hhi = sum(share * share for share in normalized_shares)
        average_firm_price = mean([firm.price for firm in firms]) if firms else 1.0
        price_dispersion = (
            math.sqrt(
                mean([(firm.price - average_firm_price) ** 2 for firm in firms])
            )
            if firms
            else 0.0
        )
        aggressive_threshold = float(
            firm_cfg.get("competition", {}).get(
                "aggressive_price_threshold", 0.985
            )
        )
        aggressive_price_market_share = sum(
            share
            for firm, share in zip(firms, normalized_shares)
            if firm.price < average_firm_price * aggressive_threshold
        )
        below_cost_pricing_market_share = sum(
            share
            for firm, share in zip(firms, normalized_shares)
            if firm.price
            < (
                firm.wage_bill + firm.fixed_cost + firm.interest_payment
            )
            / max(firm.capacity, 1e-9)
        )
        retention_eligible_wage_bill = sum(
            firm.wage_bill
            for firm in firms
            if len(firm.employee_ids)
            >= max(1, math.ceil(0.85 * max(firm.initial_employee_count, 1)))
        )
        total_wage_workers = sum(len(firm.employee_ids) for firm in firms)
        average_work_intensity = (
            sum(firm.work_intensity * len(firm.employee_ids) for firm in firms)
            / total_wage_workers
            if total_wage_workers
            else 0.0
        )
        average_required_work_hours = (
            mean(
                [
                    resident.monthly_work_hours
                    for resident in state.residents.values()
                    if resident.employed and not resident.self_employed
                ]
            )
            if wage_employed_count
            else 0.0
        )
        culture_metrics: dict[str, float | int] = {}
        culture_names = [
            str(spec["name"]) for spec in firm_cfg.get("cultures", [])
        ]
        for culture_name in culture_names:
            selected = [firm for firm in firms if firm.culture == culture_name]
            prefix = f"culture_{culture_name}"
            culture_metrics[f"{prefix}_firm_count"] = len(selected)
            culture_metrics[f"{prefix}_employment"] = sum(
                len(firm.employee_ids) for firm in selected
            )
            employment_index = sum(
                len(firm.employee_ids) for firm in selected
            ) / max(state.initial_culture_employment.get(culture_name, 0), 1)
            culture_metrics[f"{prefix}_employment_index"] = employment_index
            # Backward-compatible alias. The denominator is now the fixed
            # month-0 culture workforce, so exits cannot erase lost jobs.
            culture_metrics[f"{prefix}_employment_retention"] = employment_index
            culture_metrics[f"{prefix}_market_share"] = sum(
                firm.market_share for firm in selected
            )
            culture_metrics[f"{prefix}_sales"] = sum(firm.sales for firm in selected)
            culture_metrics[f"{prefix}_retained_profit"] = sum(
                firm.retained_profit for firm in selected
            )
            culture_metrics[f"{prefix}_average_price"] = (
                mean([firm.price for firm in selected]) if selected else 0.0
            )
        metric = {
            "month": month,
            "scenario": state.scenario,
            "population": len(state.residents),
            "firm_count": len(state.firms),
            "firm_entries": firm_entries,
            "firm_exits": firm_exits,
            "firm_exits_cash_insolvent": self._last_exit_reason_counts[
                "cash_insolvent"
            ],
            "firm_exits_operating_failure": self._last_exit_reason_counts[
                "operating_failure"
            ],
            "firm_exits_deep_contraction": self._last_exit_reason_counts[
                "deep_contraction"
            ],
            "firms_in_restructuring": sum(
                firm.restructuring_months > 0 for firm in firms
            ),
            "maximum_restructuring_months": max(
                (firm.restructuring_months for firm in firms),
                default=0,
            ),
            "entry_jobs_created": entry_jobs,
            "exit_jobs_lost": exit_jobs,
            "cumulative_firm_entries": state.cumulative_firm_entries,
            "cumulative_firm_exits": state.cumulative_firm_exits,
            "cumulative_entry_jobs": state.cumulative_entry_jobs,
            "cumulative_exit_jobs": state.cumulative_exit_jobs,
            "employment": employed_count,
            "employment_rate": employed_count / len(state.residents),
            "wage_employment": wage_employed_count,
            "wage_employment_rate": wage_employed_count / len(state.residents),
            "self_employment": self_employed_count,
            "self_employment_rate": self_employed_count / len(state.residents),
            "unemployment_rate": 1.0 - employed_count / len(state.residents),
            "long_unemployment_rate": long_unemployed / len(state.residents),
            "hires": hired,
            "fires": fired,
            "routine_layoffs": routine_layoffs,
            "firm_exit_layoffs": exit_jobs,
            "distress_exemption_layoffs": sum(
                firm.distress_exemption_layoffs for firm in firms
            ),
            "cumulative_distress_exemption_layoffs": (
                state.cumulative_distress_exemption_layoffs
            ),
            "solo_entries": solo_entries,
            "solo_exits": solo_exits,
            "voluntary_wage_exits": voluntary_wage_exits,
            "cumulative_solo_entries": state.cumulative_solo_entries,
            "cumulative_solo_exits": state.cumulative_solo_exits,
            "cumulative_voluntary_wage_exits": (
                state.cumulative_voluntary_wage_exits
            ),
            "solo_enterprise_sales": solo_enterprise_sales,
            "solo_enterprise_income": solo_enterprise_income,
            "solo_income_tax": solo_income_tax,
            "solo_substitution_sales": solo_sales_channels["substitution"],
            "solo_b2b_sales": solo_sales_channels["b2b"],
            "solo_induced_demand_sales": solo_sales_channels["induced"],
            "solo_external_sales": solo_sales_channels["external"],
            "solo_net_additional_demand": (
                solo_sales_channels["induced"]
                + solo_sales_channels["external"]
            ),
            "solo_incumbent_displacement": (
                solo_sales_channels["substitution"]
                + solo_sales_channels["b2b"]
            ),
            "cumulative_solo_substitution_sales": (
                state.cumulative_solo_substitution_sales
            ),
            "cumulative_solo_b2b_sales": state.cumulative_solo_b2b_sales,
            "cumulative_solo_induced_demand_sales": (
                state.cumulative_solo_induced_demand_sales
            ),
            "cumulative_solo_external_sales": (
                state.cumulative_solo_external_sales
            ),
            "average_work_intensity": average_work_intensity,
            "average_required_work_hours": average_required_work_hours,
            "ai_attributable_layoffs_blocked": sum(
                firm.ai_attributable_layoffs_blocked for firm in firms
            ),
            "cumulative_ai_attributable_layoffs_blocked": (
                state.cumulative_ai_attributable_layoffs_blocked
            ),
            "household_desired_consumption": planned_consumption_total,
            "household_consumption": household_consumption,
            "real_consumption": real_consumption,
            "personal_ai_spending": sum(
                resident.personal_ai_spending
                for resident in state.residents.values()
            ),
            "personal_ai_mean_use_rate": mean(
                [
                    resident.personal_ai_use_rate
                    for resident in state.residents.values()
                ]
            ),
            "government_purchase": actual_government_purchase,
            "government_real_purchase": actual_government_purchase
            / max(aggregate_price, 1e-9),
            "government_procurement": actual_procurement,
            "government_real_procurement": actual_procurement
            / max(aggregate_price, 1e-9),
            "government_employment_support_procurement": government.employment_support_procurement,
            "government_productivity_dividend_procurement": government.productivity_dividend_procurement,
            "government_retention_wage_subsidy": (
                government.retention_wage_subsidy
            ),
            "cumulative_retention_wage_subsidy": (
                government.cumulative_retention_wage_subsidy
            ),
            "government_policy_strategy": str(
                gov_cfg.get("policy_strategy", "passive_safety_net")
            ),
            "government_public_investment": actual_public_investment,
            "government_public_capital": government.public_capital,
            "firm_investment": actual_firm_investment,
            "firm_sales": actual_sales,
            "unmet_final_demand": total_orders - actual_sales,
            "aggregate_price": aggregate_price,
            "firm_average_price": average_firm_price,
            "firm_price_dispersion": price_dispersion,
            "market_hhi": market_hhi,
            "aggressive_price_market_share": aggressive_price_market_share,
            "below_cost_pricing_market_share": below_cost_pricing_market_share,
            "retention_eligible_wage_bill": retention_eligible_wage_bill,
            "total_capacity": sum(f.capacity for f in state.firms.values()),
            "unsold_output": sum(f.capacity - f.sales / max(f.price, 1e-9) for f in state.firms.values()),
            "capacity_utilization": sum(f.sales / max(f.price, 1e-9) for f in state.firms.values()) / max(sum(f.capacity for f in state.firms.values()), 1e-9),
            "gross_wage_bill": gross_wages,
            "retained_profit": retained_profit_total,
            "household_cash": sum(r.cash for r in state.residents.values()),
            "household_cash_change": sum(r.cash for r in state.residents.values()) - cash_before,
            "household_deposits": sum(r.deposits for r in state.residents.values()),
            "household_managed_fund_assets": sum(r.managed_fund_assets for r in state.residents.values()),
            "household_financial_wealth": household_financial_wealth,
            "household_saving_flow": household_saving_flow,
            "rolling_12_household_saving": rolling_12_household_saving,
            "rolling_12_financial_asset_accumulation": rolling_12_financial_asset_accumulation,
            "rolling_12_nominal_output": rolling_12_nominal_output,
            "rolling_12_macro_gdp": rolling_12_macro_gdp,
            "household_saving_to_output_ratio": rolling_12_household_saving
            / max(rolling_12_nominal_output, 1.0),
            "household_saving_to_gdp_ratio": rolling_12_household_saving
            / max(rolling_12_macro_gdp, 1.0),
            "financial_asset_accumulation_to_output_ratio": rolling_12_financial_asset_accumulation
            / max(rolling_12_nominal_output, 1.0),
            "financial_asset_accumulation_to_gdp_ratio": rolling_12_financial_asset_accumulation
            / max(rolling_12_macro_gdp, 1.0),
            "household_deposit_to_annual_income": sum(
                r.deposits for r in state.residents.values()
            ) / max(annual_household_income, 1.0),
            "household_wealth_to_annual_income": household_financial_wealth
            / max(annual_household_income, 1.0),
            "household_deposit_flow": household_deposit_flow,
            "household_managed_fund_flow": household_managed_fund_flow,
            "household_withdrawal_flow": household_withdrawal_flow,
            "firm_cash": sum(f.cash for f in state.firms.values()),
            "firm_bank_debt": sum(f.bank_debt for f in state.firms.values()),
            "firm_principal_repayment": firm_principal_repayment,
            "distressed_firm_share": sum(f.distressed_months > 0 for f in state.firms.values()) / max(len(state.firms), 1),
            "labor_tax": labor_tax,
            "indirect_tax": indirect_tax_total,
            "profit_tax": profit_tax_total,
            "ai_infrastructure_levy": ai_levy_total,
            "ai_levy_rent_base": sum(
                firm.ai_levy_rent_base for firm in firms
            ),
            "ai_levy_capture_rate": capture_rate,
            "government_tax_revenue": tax_revenue,
            "government_spending": government_spending,
            "government_fiscal_balance": government.fiscal_balance,
            "government_cash": government.cash,
            "government_debt": government.debt,
            "government_formal_debt": government.debt,
            "government_arrears": government.arrears,
            "government_arrears_incurred": government.arrears_incurred,
            "government_arrears_repayment": government.arrears_repayment,
            "government_total_liabilities": government.total_liabilities,
            "government_formal_debt_ratio": government.debt / annual_output,
            "government_arrears_ratio": government.arrears / annual_output,
            "government_debt_ratio": government.total_liabilities / annual_output,
            "government_fiscal_shortfall": government.fiscal_shortfall,
            "government_fiscal_curtailment": government.fiscal_curtailment,
            "government_statutory_funding_gap": government.statutory_funding_gap,
            "government_ai_use_rate": government.ai_use_rate,
            "government_ai_levy_fund_balance": (
                government.ai_levy_fund_balance
            ),
            "government_ai_levy_bridge_advance": (
                government.ai_levy_bridge_advance
            ),
            "government_ai_levy_public_service_spending": (
                government.ai_levy_public_service_spending
            ),
            "government_ai_levy_public_investment": (
                government.ai_levy_public_investment
            ),
            "cumulative_ai_levy_revenue": (
                government.cumulative_ai_levy_revenue
            ),
            "cumulative_ai_levy_public_service_spending": (
                government.cumulative_ai_levy_public_service_spending
            ),
            "cumulative_ai_levy_public_investment": (
                government.cumulative_ai_levy_public_investment
            ),
            "public_service_index": government.public_service_index,
            "bank_reserves": state.bank.reserves,
            "bank_liquid_assets": state.bank.reserves,
            "bank_deposits": state.bank.deposits,
            "bank_managed_funds": state.bank.managed_funds,
            "bank_total_funding": bank_funding,
            "bank_firm_loans": state.bank.firm_loans,
            "bank_government_loans": state.bank.government_loans,
            "bank_government_bonds": state.bank.government_loans,
            "bank_other_financial_assets": state.bank.other_financial_assets,
            "bank_other_asset_purchases": state.bank.other_asset_purchases,
            "bank_credit_to_firms": state.bank.firm_credit_disbursed,
            "bank_firm_credit_requested": state.bank.firm_credit_requested,
            "bank_firm_credit_rejected": state.bank.firm_credit_rejected,
            "bank_credit_to_government": state.bank.government_credit_disbursed,
            "bank_government_bond_purchases": state.bank.government_credit_disbursed,
            "bank_credit_asset_ratio": (
                state.bank.firm_loans
                + state.bank.government_loans
                + state.bank.other_financial_assets
            ) / max(bank_funding, 1.0),
            "bank_sampled_credit_ratio": (
                state.bank.firm_loans + state.bank.government_loans
            ) / max(bank_funding, 1.0),
            "bank_reserve_ratio": state.bank.reserves / max(bank_funding, 1.0),
            "bank_liquid_asset_ratio": state.bank.reserves
            / max(bank_funding, 1.0),
            "bank_loan_to_funding_ratio": (
                state.bank.firm_loans + state.bank.government_loans
            ) / max(bank_funding, 1.0),
            "bank_equity": state.bank.equity,
            "bank_provisions": state.bank.provisions,
            "bank_risk_weighted_assets": self._bank_risk_weighted_assets(),
            "bank_capital_adequacy_ratio": bank_capital_ratio,
            "bank_nonperforming_loans": nonperforming_loans,
            "bank_npl_ratio": nonperforming_loans
            / max(state.bank.firm_loans, 1.0),
            "bank_provision_coverage_ratio": (
                state.bank.provisions / nonperforming_loans
                if nonperforming_loans > 0.0
                else 0.0
            ),
            "bank_interest_income": state.bank.interest_income,
            "bank_interest_expense": state.bank.interest_expense,
            "bank_net_interest_income": state.bank.interest_income
            - state.bank.interest_expense,
            "bank_provision_expense": state.bank.provision_expense,
            "bank_writeoffs": state.bank.writeoffs,
            "bank_recoveries": state.bank.recoveries,
            "firm_loans_to_annual_output": state.bank.firm_loans
            / max(annual_output, 1.0),
            "firm_loans_to_macro_gdp": state.bank.firm_loans
            / max(rolling_12_macro_gdp, 1.0),
            "government_bonds_to_annual_output": state.bank.government_loans
            / max(annual_output, 1.0),
            "government_bonds_to_macro_gdp": state.bank.government_loans
            / max(rolling_12_macro_gdp, 1.0),
            "rolling_12_net_firm_credit": rolling_12_net_firm_credit,
            "rolling_12_net_government_bonds": rolling_12_net_government_bonds,
            "net_firm_credit_to_output_ratio": rolling_12_net_firm_credit
            / max(rolling_12_nominal_output, 1.0),
            "net_firm_credit_to_gdp_ratio": rolling_12_net_firm_credit
            / max(rolling_12_macro_gdp, 1.0),
            "net_government_bonds_to_output_ratio": rolling_12_net_government_bonds
            / max(rolling_12_nominal_output, 1.0),
            "net_government_bonds_to_gdp_ratio": rolling_12_net_government_bonds
            / max(rolling_12_macro_gdp, 1.0),
            "bank_balance_sheet_error": state.bank.balance_sheet_error(),
            "sales_identity_error": (
                actual_sales
                - household_consumption
                - actual_government_purchase
                - actual_firm_investment
                - solo_sales_channels["external"] * fulfillment
            ),
            "wage_identity_error": gross_wages - settled_wage_bill,
            "tax_identity_error": (
                tax_revenue
                - labor_tax
                - indirect_tax_total
                - profit_tax_total
                - solo_income_tax
                - ai_levy_total
            ),
            **culture_metrics,
            **distribution_metrics,
        }
        state.month = month
        state.history.append(metric)
        state.intents.clear()
        return metric

    def _apply_intents(self) -> None:
        for resident in self.state.residents.values():
            intent = bounded_intent("resident", self.state.intents.get(resident.id))
            resident.consumption_stance = intent["consumption_stance"]
        for firm in self.state.firms.values():
            intent = bounded_intent("firm", self.state.intents.get(firm.id))
            firm.labor_stance = intent["labor_stance"]
        intent = bounded_intent("government", self.state.intents.get(self.state.government.id))
        self.state.government.policy_stance = intent["policy_stance"]

    def _update_ai(self, month: int) -> None:
        firm_cfg = self.config["firms"]
        gov_cfg = self.config["government"]
        scenario = self.config["scenario"]
        shock_month = int(self.config["simulation"]["shock_month"])
        active = month >= shock_month
        for firm in self.state.firms.values():
            target = firm.ai_target if active and scenario["private_ai"] else 1.0
            firm.ai_multiplier += float(firm_cfg["ai_adoption_speed"]) * (target - firm.ai_multiplier)
            culture = self._culture_spec(firm)
            pass_through = float(firm_cfg["price_productivity_pass_through"]) * float(
                culture.get("price_aggressiveness", 1.0)
            )
            no_levy_target_price = max(
                float(firm_cfg["minimum_price"]),
                1.0 - pass_through * (firm.ai_multiplier - 1.0),
            )
            firm.ai_levy_per_unit = 0.0
            target_price = no_levy_target_price
            if active and scenario.get("ai_infrastructure_levy", False):
                levy_cfg = self._institutional_config("ai_infrastructure_levy")
                capture_rate = self._levy_capture_rate(month)
                taxable_share = 1.0 - float(
                    levy_cfg.get("basic_consumption_exemption_share", 0.0)
                )
                productivity_rent_per_unit = max(
                    1.0 - 1.0 / max(firm.ai_multiplier, 1.0), 0.0
                )
                firm.ai_levy_per_unit = (
                    capture_rate
                    * taxable_share
                    * productivity_rent_per_unit
                )
                target_price += (
                    float(levy_cfg.get("consumer_price_pass_through", 1.0))
                    * firm.ai_levy_per_unit
                )
            firm.price += float(firm_cfg["price_adjustment_speed"]) * (target_price - firm.price)
            personal_ai = self.config["households"].get("personal_ai", {})
            targets = personal_ai.get("target_by_culture", {})
            for resident_id in firm.employee_ids:
                resident = self.state.residents[resident_id]
                cash_floor = (
                    float(personal_ai.get("minimum_cash_buffer_months", 1.0))
                    * resident.minimum_consumption
                )
                can_afford = resident.cash + resident.deposits >= cash_floor
                personal_target = (
                    float(targets.get(firm.culture, 0.0))
                    if active
                    and scenario["private_ai"]
                    and bool(personal_ai.get("enabled", False))
                    and can_afford
                    else 0.0
                )
                resident.personal_ai_use_rate += float(
                    personal_ai.get("adoption_speed", 0.0)
                ) * (personal_target - resident.personal_ai_use_rate)
                resident.personal_ai_productivity_gain = float(
                    personal_ai.get("productivity_gain", 0.0)
                )
        employed_ids = {
            resident_id
            for firm in self.state.firms.values()
            for resident_id in firm.employee_ids
        }
        personal_speed = float(
            self.config["households"].get("personal_ai", {}).get(
                "adoption_speed", 0.0
            )
        )
        solo_cfg = self._institutional_config("solo_enterprise")
        for resident in self.state.residents.values():
            if resident.id in employed_ids:
                continue
            if (
                active
                and scenario.get("solo_enterprise", False)
                and resident.self_employed
            ):
                solo_target = float(
                    solo_cfg.get("self_employed_ai_target", 0.80)
                )
                resident.personal_ai_use_rate += personal_speed * (
                    solo_target - resident.personal_ai_use_rate
                )
                resident.personal_ai_productivity_gain = float(
                    self.config["households"]
                    .get("personal_ai", {})
                    .get("productivity_gain", 0.0)
                )
            else:
                resident.personal_ai_use_rate += personal_speed * (
                    0.0 - resident.personal_ai_use_rate
                )
        government = self.state.government
        if active and scenario["government_ai"]:
            fiscal_room = government.total_liabilities < float(gov_cfg["debt_limit_ratio"]) * 12.0 * self.state.baseline_total_output
            target = 0.70 if fiscal_room else government.ai_use_rate
        else:
            target = float(gov_cfg["ai_use_rate"])
        delta = max(-float(gov_cfg["ai_adoption_speed"]), min(float(gov_cfg["ai_adoption_speed"]), target - government.ai_use_rate))
        government.ai_use_rate += delta

    def _process_solo_enterprise_transitions(
        self, month: int
    ) -> tuple[int, int, int]:
        """Incubate, form, and close one-person AI-enabled enterprises."""
        state = self.state
        for resident in state.residents.values():
            resident.solo_business_revenue = 0.0
            resident.solo_business_income = 0.0
            resident.saved_ai_hours = 0.0
        if not (
            month >= int(self.config["simulation"]["shock_month"])
            and self.config["scenario"].get("solo_enterprise", False)
        ):
            return 0, 0, 0

        cfg = self._institutional_config("solo_enterprise")
        monthly_hours = float(self.config["simulation"]["monthly_work_hours"])
        exits = 0
        for resident in state.residents.values():
            if not resident.self_employed:
                continue
            if resident.solo_loss_months < int(cfg.get("failure_months", 6)):
                resident.solo_business_months += 1
                continue
            resident.self_employed = False
            resident.employed = False
            resident.shock_unemployed = True
            resident.unemployment_duration = 0
            resident.shock_unemployment_duration = 0
            resident.monthly_work_hours = 0.0
            resident.solo_business_months = 0
            resident.solo_loss_months = 0
            exits += 1

        wage_employed = [
            resident
            for resident in state.residents.values()
            if resident.employed
            and not resident.self_employed
            and resident.employer_id is not None
        ]
        for resident in wage_employed:
            effective_gain = (
                resident.personal_ai_productivity_gain
                * resident.personal_ai_use_rate
            )
            resident.saved_ai_hours = monthly_hours * (
                1.0 - 1.0 / max(1.0 + effective_gain, 1e-9)
            )
            resident.entrepreneurial_readiness += (
                resident.saved_ai_hours / max(monthly_hours, 1.0)
            )

        maximum_self_employed = max(
            1,
            math.floor(
                len(state.residents)
                * float(cfg.get("max_self_employed_share", 0.10))
            ),
        )
        current_self_employed = sum(
            resident.self_employed for resident in state.residents.values()
        )
        available_slots = max(maximum_self_employed - current_self_employed, 0)
        monthly_limit = max(
            1,
            math.ceil(
                len(state.residents)
                * float(cfg.get("max_monthly_entry_share", 0.01))
            ),
        )
        candidates = []
        for resident in wage_employed:
            startup_cost = (
                float(cfg.get("startup_cost_months", 0.50))
                * resident.minimum_consumption
            )
            if (
                resident.entrepreneurial_readiness
                >= float(cfg.get("incubation_readiness_threshold", 0.18))
                and resident.personal_ai_use_rate
                >= float(cfg.get("minimum_personal_ai_use_rate", 0.20))
                and resident.cash >= startup_cost
            ):
                candidates.append(
                    (
                        resident.entrepreneurial_readiness,
                        resident.base_productivity,
                        -resident.id,
                        startup_cost,
                        resident,
                    )
                )
        candidates.sort(reverse=True, key=lambda item: item[:3])

        entries = 0
        voluntary_exits = 0
        for _, _, _, startup_cost, resident in candidates[
            : min(monthly_limit, available_slots)
        ]:
            employer = state.firms.get(resident.employer_id)
            if employer is None or resident.id not in employer.employee_ids:
                continue
            employer.employee_ids.remove(resident.id)
            resident.employer_id = None
            resident.gross_wage = 0.0
            resident.self_employed = True
            resident.employed = True
            resident.shock_unemployed = False
            resident.unemployment_duration = 0
            resident.shock_unemployment_duration = 0
            resident.monthly_work_hours = monthly_hours
            resident.cash -= startup_cost
            resident.solo_business_months = 1
            resident.solo_loss_months = 0
            entries += 1
            voluntary_exits += 1

        state.cumulative_solo_entries += entries
        state.cumulative_solo_exits += exits
        state.cumulative_voluntary_wage_exits += voluntary_exits
        return entries, exits, voluntary_exits

    def _plan_solo_demand(
        self,
        household_orders: float,
        government_orders: float,
        firm_investment_orders: float,
    ) -> dict[str, float]:
        """Build four separately auditable solo-enterprise demand channels."""
        zero = {
            "substitution": 0.0,
            "b2b": 0.0,
            "induced": 0.0,
            "external": 0.0,
        }
        owners = [
            resident
            for resident in self.state.residents.values()
            if resident.self_employed
        ]
        if not owners or not self.config["scenario"].get("solo_enterprise", False):
            return zero
        cfg = self._institutional_config("solo_enterprise")
        maximum_owners = max(
            1,
            math.floor(
                len(self.state.residents)
                * float(cfg.get("max_self_employed_share", 0.10))
            ),
        )
        activation = min(len(owners) / maximum_owners, 1.0)
        substitution = (
            max(household_orders + government_orders, 0.0)
            * float(
                cfg.get(
                    "substitution_market_share",
                    cfg.get("solo_market_share", 0.05),
                )
            )
            * activation
        )
        b2b = (
            max(firm_investment_orders, 0.0)
            * float(cfg.get("b2b_investment_share", 0.0))
            * activation
        )
        induced_target = (
            self.state.baseline_household_demand
            * float(cfg.get("induced_demand_baseline_share", 0.0))
            * activation
        )
        available_drawdown: list[tuple[Resident, float]] = []
        for resident in self.state.residents.values():
            excess = max(
                resident.cash
                + resident.disposable_income
                - resident.nominal_consumption
                - resident.target_cash,
                0.0,
            )
            if excess > 0.0:
                available_drawdown.append((resident, excess))
        drawdown_capacity = (
            sum(amount for _, amount in available_drawdown)
            * float(cfg.get("induced_cash_drawdown_rate", 0.0))
        )
        induced = min(induced_target, drawdown_capacity)
        if induced > 0.0 and available_drawdown:
            total_excess = sum(amount for _, amount in available_drawdown)
            for resident, amount in available_drawdown:
                resident.nominal_consumption += (
                    induced * amount / max(total_excess, 1e-9)
                )
        external = (
            self.state.baseline_household_demand
            * float(cfg.get("external_demand_baseline_share", 0.0))
            * activation
        )
        return {
            "substitution": substitution,
            "b2b": b2b,
            "induced": induced,
            "external": external,
        }

    def _allocate_solo_enterprise_sales(
        self, demand: dict[str, float]
    ) -> tuple[dict[str, float], float, float]:
        """Allocate capacity over substitution, B2B, induced, and external demand."""
        owners = [
            resident
            for resident in self.state.residents.values()
            if resident.self_employed
        ]
        zero = {key: 0.0 for key in ("substitution", "b2b", "induced", "external")}
        if not owners or not self.config["scenario"].get("solo_enterprise", False):
            return zero, 0.0, 0.0
        cfg = self._institutional_config("solo_enterprise")
        capacities = [
            max(
                resident.monthly_disposable_anchor,
                resident.minimum_consumption,
            )
            * (1.0 + resident.personal_ai_productivity_gain * resident.personal_ai_use_rate)
            for resident in owners
        ]
        desired_total = sum(max(float(value), 0.0) for value in demand.values())
        target = min(desired_total, sum(capacities))
        if target <= 0.0:
            return zero, 0.0, 0.0
        channel_scale = target / max(desired_total, 1e-9)
        realized = {
            key: max(float(demand.get(key, 0.0)), 0.0) * channel_scale
            for key in zero
        }
        total_capacity = sum(capacities)
        operating_cost_rate = float(cfg.get("operating_cost_rate", 0.25))
        tax_rate = float(cfg.get("income_tax_rate", 0.05))
        tax_total = 0.0
        net_income_total = 0.0
        for resident, capacity in zip(owners, capacities):
            revenue = target * capacity / max(total_capacity, 1e-9)
            pre_tax_income = revenue * (1.0 - operating_cost_rate)
            tax = max(pre_tax_income, 0.0) * tax_rate
            net_income = pre_tax_income - tax
            resident.solo_business_revenue = revenue
            resident.solo_business_income = net_income
            resident.cumulative_solo_business_revenue += revenue
            resident.cumulative_solo_business_income += net_income
            resident.disposable_income += net_income
            if net_income < (
                float(cfg.get("minimum_income_ratio", 0.35))
                * resident.baseline_disposable_income
            ):
                resident.solo_loss_months += 1
            else:
                resident.solo_loss_months = 0
            tax_total += tax
            net_income_total += net_income
        self.state.cumulative_solo_substitution_sales += realized[
            "substitution"
        ]
        self.state.cumulative_solo_b2b_sales += realized["b2b"]
        self.state.cumulative_solo_induced_demand_sales += realized["induced"]
        self.state.cumulative_solo_external_sales += realized["external"]
        return realized, tax_total, net_income_total

    def _government_plan(self, lagged: dict[str, Any]) -> dict[str, float]:
        state = self.state
        gov_cfg = self.config["government"]
        scenario = self.config["scenario"]
        government = state.government
        policy_active = self._policy_is_active()
        strategy = self._government_strategy()
        base_service = state.baseline_public_service or (
            state.baseline_household_demand * float(gov_cfg["public_service_share"])
        )
        base_procurement = state.baseline_procurement or (
            state.baseline_household_demand * float(gov_cfg["base_procurement_share"])
        )
        consumption_gap = max(
            state.baseline_household_demand
            - float(
                lagged.get(
                    "household_consumption", state.baseline_household_demand
                )
            ),
            0.0,
        )
        unemployment_gap = max(
            float(lagged.get("unemployment_rate", 0.0))
            - float(gov_cfg["target_unemployment_rate"]),
            0.0,
        )
        desired_regular_procurement = base_procurement
        desired_employment_support = 0.0
        desired_productivity_dividend = 0.0
        if policy_active and scenario["procurement_response"]:
            desired_regular_procurement += (
                float(gov_cfg["procurement_response_rate"])
                * float(strategy.get("procurement_response_multiplier", 1.0))
                * consumption_gap
            )
            average_ai_gain = mean(
                [max(firm.ai_multiplier - 1.0, 0.0) for firm in state.firms.values()]
                or [0.0]
            )
            ai_activation = min(average_ai_gain / 0.10, 1.0)
            maintained_wage_bill = float(
                lagged.get(
                    "retention_eligible_wage_bill",
                    sum(
                        firm.wage_bill
                        for firm in state.firms.values()
                        if len(firm.employee_ids)
                        >= max(
                            1,
                            math.ceil(
                                0.85 * max(firm.initial_employee_count, 1)
                            ),
                        )
                    ),
                )
            )
            demand_stress = min(
                max(
                    consumption_gap / max(state.baseline_household_demand, 1.0),
                    unemployment_gap / 0.05,
                ),
                1.0,
            )
            months_since_shock = max(
                state.month + 1 - int(self.config["simulation"]["shock_month"]),
                0,
            )
            transition_support = ai_activation * max(
                1.0 - months_since_shock / 24.0, 0.0
            )
            support_activation = max(demand_stress, 0.50 * transition_support)
            desired_employment_support = (
                float(strategy.get("employment_support_rate", 0.0))
                * maintained_wage_bill
                * support_activation
            )
            lagged_price = max(float(lagged.get("aggregate_price", 1.0)), 1e-9)
            purchasing_power_gain = max(1.0 / lagged_price - 1.0, 0.0)
            desired_productivity_dividend = (
                float(strategy.get("productivity_dividend_rate", 0.0))
                * base_procurement
                * purchasing_power_gain
            )
        speed = float(gov_cfg["policy_adjustment_speed"])
        previous_support = government.employment_support_procurement
        previous_dividend = government.productivity_dividend_procurement
        previous_regular = max(
            government.procurement - previous_support - previous_dividend, 0.0
        )
        regular_procurement = previous_regular + speed * (
            desired_regular_procurement - previous_regular
        )
        employment_support = previous_support + speed * (
            desired_employment_support - previous_support
        )
        productivity_dividend = previous_dividend + speed * (
            desired_productivity_dividend - previous_dividend
        )
        procurement = (
            regular_procurement + employment_support + productivity_dividend
        )
        stance_multiplier = 1.0
        if government.policy_stance == "stabilize":
            stance_multiplier = 1.05
        elif government.policy_stance == "balanced_support":
            stance_multiplier = 1.025
        elif government.policy_stance == "fiscal_guard":
            stance_multiplier = 0.90
        regular_procurement *= stance_multiplier
        employment_support *= stance_multiplier
        productivity_dividend *= stance_multiplier
        procurement = max(
            base_procurement if government.policy_stance == "fiscal_guard" else 0.0,
            regular_procurement + employment_support + productivity_dividend,
        )
        mean_wage = mean([r.last_net_wage for r in state.residents.values() if r.last_net_wage > 0] or [0.0])
        extra_pool = 0.0
        if policy_active and scenario["transfer_response"]:
            extra_pool = (
                float(gov_cfg["transfer_response_rate"])
                * float(strategy.get("transfer_response_multiplier", 1.0))
                * unemployment_gap
                * len(state.residents)
                * mean_wage
            )
        ai_spending = (
            state.baseline_ai_spending
            * government.ai_use_rate
            / max(float(gov_cfg["ai_use_rate"]), 1e-12)
            if state.baseline_ai_spending
            else base_service
            * float(gov_cfg["ai_budget_share_of_service"])
            * government.ai_use_rate
        )
        public_investment = 0.0
        if self._banking_active():
            public_investment = self.state.baseline_total_output * float(
                self.config["banking"].get(
                    "public_investment_share_of_output", 0.0
                )
            )
        levy_public_service = 0.0
        levy_public_investment = 0.0
        retention_wage_subsidy = sum(
            self._retention_subsidy_by_firm.values()
        )
        if policy_active and scenario.get("ai_infrastructure_levy", False):
            levy_cfg = self._institutional_config("ai_infrastructure_levy")
            forecast_levy = self._forecast_ai_levy(self.state.month + 1)
            available_levy_fund = max(
                government.ai_levy_fund_balance
                + float(levy_cfg.get("same_month_advance_share", 0.0))
                * forecast_levy,
                0.0,
            )
            service_share = float(levy_cfg.get("public_service_share", 0.70))
            investment_share = float(
                levy_cfg.get("public_investment_share", 0.30)
            )
            allocation_share = service_share + investment_share
            if allocation_share > 1.0 + 1e-9:
                raise ValueError(
                    "AI levy public-service and public-investment shares exceed 1"
                )
            levy_public_service = available_levy_fund * service_share
            levy_public_investment = available_levy_fund * investment_share
            base_service += levy_public_service
            public_investment += levy_public_investment
        return {
            "public_service": base_service,
            "procurement": procurement,
            "ai_spending": ai_spending,
            "public_investment": public_investment,
            "transfer_extra_pool": extra_pool,
            "employment_support_procurement": employment_support,
            "productivity_dividend_procurement": productivity_dividend,
            "ai_levy_public_service": levy_public_service,
            "ai_levy_public_investment": levy_public_investment,
            "retention_wage_subsidy": retention_wage_subsidy,
        }

    def _resident_transfer(self, resident: Resident, extra_pool: float) -> float:
        if resident.employed:
            return 0.0
        gov_cfg = self.config["government"]
        base = resident.baseline_transfer or max(
            float(gov_cfg["minimum_unemployment_transfer"]),
            float(gov_cfg["unemployment_replacement_rate"]) * resident.last_net_wage,
        )
        unemployed = [r for r in self.state.residents.values() if not r.employed]
        if not unemployed:
            return 0.0
        vulnerability = 1.0 / max(resident.monthly_disposable_anchor, 1.0)
        denominator = sum(1.0 / max(r.monthly_disposable_anchor, 1.0) for r in unemployed)
        return base * float(getattr(self, "_base_transfer_scale", 1.0)) + extra_pool * vulnerability / denominator

    def _apply_fiscal_limit(self, plan: dict[str, float]) -> dict[str, float]:
        government = self.state.government
        gov_cfg = self.config["government"]
        base_procurement = self.state.baseline_procurement or (
            self.state.baseline_household_demand
            * float(gov_cfg["base_procurement_share"])
        )
        base_service = self.state.baseline_public_service or (
            self.state.baseline_household_demand
            * float(gov_cfg["public_service_share"])
        )
        service_floor = base_service * float(gov_cfg.get("public_service_floor_ratio", 0.50))
        monthly_cap = self._government_parameter("max_annual_deficit_ratio") * self.state.baseline_total_output
        debt_limit = self._government_parameter("debt_limit_ratio") * 12.0 * self.state.baseline_total_output
        borrowing_room = min(
            monthly_cap,
            max(debt_limit - government.total_liabilities, 0.0),
        )
        expected_tax = 0.95 * max(government.tax_revenue, 0.0)
        available = max(government.cash, 0.0) + expected_tax + borrowing_room
        unemployed = [resident for resident in self.state.residents.values() if not resident.employed]
        base_transfers = sum(
            resident.baseline_transfer
            or max(
                float(gov_cfg["minimum_unemployment_transfer"]),
                float(gov_cfg["unemployment_replacement_rate"]) * resident.last_net_wage,
            )
            for resident in unemployed
        )
        components = {
            "base_transfers": base_transfers,
            "extra_transfers": plan["transfer_extra_pool"],
            "retention_wage_subsidy": plan.get(
                "retention_wage_subsidy", 0.0
            ),
            "protected_public_service": min(plan["public_service"], service_floor),
            "discretionary_public_service": max(plan["public_service"] - service_floor, 0.0),
            "ai_spending": plan["ai_spending"],
            "public_investment": plan.get("public_investment", 0.0),
            "base_procurement": min(plan["procurement"], base_procurement),
            "extra_procurement": max(plan["procurement"] - base_procurement, 0.0),
        }
        planned_total = sum(components.values())
        excess = max(planned_total - available, 0.0)
        # Optional stabilization is cut before recurring commitments. Base
        # transfers and the statutory service floor remain payable; any
        # residual funding gap is surfaced rather than erasing the channel.
        strategy = self._government_strategy()
        cut_priority = strategy.get(
            "cut_priority",
            (
                "extra_procurement",
                "extra_transfers",
                "public_investment",
                "base_procurement",
                "ai_spending",
                "discretionary_public_service",
            ),
        )
        for key in cut_priority:
            if key not in components or key in {"base_transfers", "protected_public_service"}:
                continue
            cut = min(components[key], excess)
            components[key] -= cut
            excess -= cut
            if excess <= 1e-9:
                break
        government.fiscal_curtailment = planned_total - sum(components.values())
        government.statutory_funding_gap = max(excess, 0.0)
        self._base_transfer_scale = components["base_transfers"] / base_transfers if base_transfers else 1.0
        final_procurement = components["base_procurement"] + components["extra_procurement"]
        planned_procurement = max(plan["procurement"], 1e-9)
        procurement_scale = min(final_procurement / planned_procurement, 1.0)
        public_service_scale = min(
            (
                components["protected_public_service"]
                + components["discretionary_public_service"]
            )
            / max(plan["public_service"], 1e-9),
            1.0,
        )
        public_investment_scale = min(
            components["public_investment"]
            / max(plan.get("public_investment", 0.0), 1e-9),
            1.0,
        )
        return {
            "public_service": components["protected_public_service"] + components["discretionary_public_service"],
            "procurement": final_procurement,
            "ai_spending": components["ai_spending"],
            "public_investment": components["public_investment"],
            "transfer_extra_pool": components["extra_transfers"],
            "retention_wage_subsidy": components[
                "retention_wage_subsidy"
            ],
            "employment_support_procurement": plan.get(
                "employment_support_procurement", 0.0
            )
            * procurement_scale,
            "productivity_dividend_procurement": plan.get(
                "productivity_dividend_procurement", 0.0
            )
            * procurement_scale,
            "ai_levy_public_service": plan.get(
                "ai_levy_public_service", 0.0
            )
            * public_service_scale,
            "ai_levy_public_investment": plan.get(
                "ai_levy_public_investment", 0.0
            )
            * public_investment_scale,
        }

    def _settle_government(self, tax_revenue: float, spending: float) -> None:
        government = self.state.government
        bank = self.state.bank
        government.fiscal_shortfall = 0.0
        government.arrears_incurred = 0.0
        government.arrears_repayment = 0.0
        balance = tax_revenue - spending
        if balance >= 0:
            arrears_repayment = min(government.arrears, balance)
            government.arrears -= arrears_repayment
            government.arrears_repayment = arrears_repayment
            balance -= arrears_repayment

            debt_repayment = min(government.debt, balance)
            government.debt -= debt_repayment
            if self._banking_enabled():
                bank_repayment = min(debt_repayment, bank.government_loans)
                bank.government_loans -= bank_repayment
                bank.reserves += bank_repayment
                bank.principal_repayments += bank_repayment
            government.cash += balance - debt_repayment
            return
        government.cash += balance
        if government.cash >= 0:
            return
        need = -government.cash
        monthly_cap = self._government_parameter("max_annual_deficit_ratio") * self.state.baseline_total_output
        debt_limit = self._government_parameter("debt_limit_ratio") * 12.0 * self.state.baseline_total_output
        borrowing = min(
            need,
            monthly_cap,
            max(debt_limit - government.total_liabilities, 0.0),
        )
        if self._banking_enabled():
            funding = bank.deposits + bank.managed_funds
            maximum_bonds = funding * float(
                self.config["banking"].get(
                    "max_government_bond_share",
                    self.config["banking"].get(
                        "target_government_bond_share", 1.0
                    ),
                )
            )
            bond_room = max(maximum_bonds - bank.government_loans, 0.0)
            total_credit_room = max(
                funding
                * float(
                    self.config["banking"].get(
                        "maximum_loan_to_funding_ratio", 0.80
                    )
                )
                - bank.firm_loans
                - bank.government_loans,
                0.0,
            )
            borrowing = min(
                borrowing,
                self._bank_available_credit(),
                bond_room,
                total_credit_room,
                self._capital_credit_room(
                    float(
                        self.config["banking"].get(
                            "government_bond_risk_weight", 0.20
                        )
                    )
                ),
            )
            bank.reserves -= borrowing
            bank.government_loans += borrowing
            bank.government_credit_disbursed += borrowing
        government.debt += borrowing
        government.cash += borrowing
        government.fiscal_shortfall = max(-government.cash, 0.0)
        government.arrears_incurred = government.fiscal_shortfall
        government.arrears += government.arrears_incurred
        government.cash = max(government.cash, 0.0)

    def _liquidate_firm_debt(self, firm: Firm) -> None:
        """Close a failed firm's loan and recognize recoveries and losses."""
        exposure = max(firm.bank_debt, 0.0)
        if exposure <= 0.0:
            return
        bank = self.state.bank
        loss_given_default = float(
            self.config.get("banking", {}).get("loss_given_default", 0.45)
        )
        recoverable_assets = max(firm.cash, 0.0) + max(
            firm.investment_capital, 0.0
        ) * max(1.0 - loss_given_default, 0.0)
        recovery = min(exposure, recoverable_assets)
        provision_used = min(bank.provisions, exposure)
        bank.firm_loans = max(bank.firm_loans - exposure, 0.0)
        bank.provisions = max(bank.provisions - provision_used, 0.0)
        bank.reserves += recovery
        bank.equity += recovery - exposure + provision_used
        bank.writeoffs += exposure
        bank.recoveries += recovery
        firm.loan_writeoff = exposure
        firm.collateral_recovery = recovery
        firm.bank_debt = 0.0

    def _process_firm_exits(self, month: int) -> tuple[int, int]:
        """Liquidate persistently insolvent or deeply contracted firms."""
        cfg = self.config["firms"]
        state = self.state
        self._last_exit_reason_counts = {
            "cash_insolvent": 0,
            "operating_failure": 0,
            "deep_contraction": 0,
        }
        if not bool(cfg.get("enable_entry_exit", False)) or len(state.firms) <= 1:
            return 0, 0
        minimum_age = int(cfg.get("bankruptcy_minimum_age_months", 12))
        cash_months = int(cfg.get("bankruptcy_cash_distress_months", 6))
        loss_months = int(cfg.get("bankruptcy_loss_months", 6))
        employment_ratio = float(cfg.get("bankruptcy_employment_ratio", 0.35))
        responsibility_active = self._institution_is_active(
            "employment_responsibility"
        )
        responsibility_cfg = self._institutional_config(
            "employment_responsibility"
        )
        restructuring_grace = int(
            responsibility_cfg.get("restructuring_grace_months", 0)
        )
        candidates: list[
            tuple[tuple[int, int, float], Firm, dict[str, bool]]
        ] = []
        for firm in state.firms.values():
            age = month - firm.founding_month
            if age < minimum_age:
                continue
            workforce_anchor = max(
                firm.peak_employee_count,
                firm.initial_employee_count,
                1,
            )
            deeply_contracted = (
                len(firm.employee_ids) / workforce_anchor <= employment_ratio
            )
            cash_insolvent = firm.distressed_months >= cash_months
            operating_failure = firm.loss_months >= loss_months
            if cash_insolvent or operating_failure or deeply_contracted:
                if (
                    responsibility_active
                    and firm.restructuring_months < restructuring_grace
                ):
                    firm.restructuring_months += 1
                    continue
                severity = (
                    int(cash_insolvent),
                    firm.loss_months,
                    -firm.cash,
                )
                candidates.append(
                    (
                        severity,
                        firm,
                        {
                            "cash_insolvent": cash_insolvent,
                            "operating_failure": operating_failure,
                            "deep_contraction": deeply_contracted,
                        },
                    )
                )
            else:
                firm.restructuring_months = 0
        candidates.sort(key=lambda item: item[0], reverse=True)
        limit = min(
            int(cfg.get("max_monthly_exits", 2)),
            max(len(state.firms) - 1, 0),
        )
        jobs_lost = 0
        exited = 0
        for _, firm, reasons in candidates[:limit]:
            for reason, active in reasons.items():
                if active:
                    self._last_exit_reason_counts[reason] += 1
            jobs_lost += len(firm.employee_ids)
            for resident_id in list(firm.employee_ids):
                resident = state.residents[resident_id]
                resident.employed = False
                resident.shock_unemployed = True
                resident.shock_unemployment_duration = 0
                resident.monthly_work_hours = 0.0
                resident.employer_id = None
                resident.gross_wage = 0.0
            firm.employee_ids.clear()
            self._liquidate_firm_debt(firm)
            state.intents.pop(firm.id, None)
            del state.firms[firm.id]
            exited += 1
        if exited:
            state.cumulative_firm_exits += exited
            state.cumulative_exit_jobs += jobs_lost
            self._update_loan_provisions()
        return exited, jobs_lost

    def _entry_type_spec(self) -> dict[str, Any]:
        """Choose the most underrepresented configured firm type."""
        type_specs = list(self.config["firms"]["types"])
        counts = {
            str(spec["name"]): sum(
                firm.firm_type == str(spec["name"])
                for firm in self.state.firms.values()
            )
            for spec in type_specs
        }
        next_total = len(self.state.firms) + 1
        return max(
            type_specs,
            key=lambda spec: (
                float(spec["share"])
                - (counts[str(spec["name"])] + 1) / next_total,
                -counts[str(spec["name"])],
            ),
        )

    def _entry_culture(self) -> str:
        cfg = self.config["firms"]
        mode = str(cfg.get("culture_mode", "mixed"))
        if mode != "mixed":
            return mode
        specs = list(cfg.get("cultures", [{"name": "adaptive", "share": 1.0}]))
        counts = {
            str(spec["name"]): sum(
                firm.culture == str(spec["name"])
                for firm in self.state.firms.values()
            )
            for spec in specs
        }
        next_total = len(self.state.firms) + 1
        selected = max(
            specs,
            key=lambda spec: (
                float(spec.get("share", 0.0))
                - (counts[str(spec["name"])] + 1) / next_total,
                -counts[str(spec["name"])],
            ),
        )
        return str(selected["name"])

    def _process_firm_entries(
        self,
        month: int,
        unmet_demand_ratio: float,
        immediate: bool = False,
    ) -> tuple[int, int]:
        """Finance viable entrants and let them recruit from unemployment."""
        cfg = self.config["firms"]
        state = self.state
        if not bool(cfg.get("enable_entry_exit", False)):
            return 0, 0
        shock_month = int(self.config["simulation"]["shock_month"])
        interval = max(int(cfg.get("entry_interval_months", 3)), 1)
        if month < shock_month or (
            not immediate and (month - shock_month) % interval != 0
        ):
            return 0, 0
        maximum_firms = max(
            state.initial_firm_count,
            round(
                state.initial_firm_count
                * float(cfg.get("max_firm_count_multiplier", 2.0))
            ),
        )
        if len(state.firms) >= maximum_firms or not self._banking_active(month):
            return 0, 0
        unemployed = sorted(
            (resident for resident in state.residents.values() if not resident.employed),
            key=lambda resident: resident.base_productivity,
            reverse=True,
        )
        unemployment_rate = len(unemployed) / max(len(state.residents), 1)
        if unemployment_rate < float(
            cfg.get("entry_unemployment_threshold", 0.01)
        ):
            return 0, 0
        average_ai_gain = mean(
            [max(firm.ai_multiplier - 1.0, 0.0) for firm in state.firms.values()]
        )
        ai_opportunity = average_ai_gain >= float(
            cfg.get("entry_ai_opportunity_threshold", 0.02)
        )
        demand_opportunity = unmet_demand_ratio >= float(
            cfg.get("entry_unmet_demand_threshold", 0.01)
        )
        if not (ai_opportunity or demand_opportunity):
            return 0, 0

        jobs_per_firm = max(
            1,
            round(
                len(state.residents)
                / 500.0
                * float(cfg.get("startup_employees_per_500", 4.0))
            ),
        )
        entry_limit = min(
            int(cfg.get("max_monthly_entries", 1)),
            maximum_firms - len(state.firms),
            len(unemployed) // jobs_per_firm,
        )
        entries = 0
        jobs_created = 0
        for _ in range(entry_limit):
            recruits = unemployed[:jobs_per_firm]
            if len(recruits) < jobs_per_firm:
                break
            type_spec = self._entry_type_spec()
            culture = self._entry_culture()
            culture_spec = next(
                (
                    spec
                    for spec in cfg.get("cultures", [])
                    if str(spec.get("name")) == culture
                ),
                {},
            )
            private_ai = bool(self.config["scenario"].get("private_ai", False))
            ai_target = float(type_spec["ai_target"])
            ai_multiplier = 1.0
            price = max(
                float(cfg["minimum_price"]),
                1.0
                - float(cfg["price_productivity_pass_through"])
                * float(culture_spec.get("price_aggressiveness", 1.0))
                * (ai_multiplier - 1.0),
            )
            capacity = (
                state.productivity_scale
                * ai_multiplier
                * sum(resident.base_productivity for resident in recruits)
            )
            expected_sales = capacity * float(cfg["target_utilization"]) * price
            wage_factor = 1.0 + float(cfg["wage_productivity_pass_through"]) * (
                ai_multiplier - 1.0
            )
            wage_bill = sum(
                resident.baseline_gross_wage * wage_factor for resident in recruits
            )
            indirect_tax = expected_sales * float(
                self.config["government"]["indirect_tax_rate"]
            )
            target_profit = expected_sales * float(cfg["target_pre_tax_margin"])
            fixed_cost = max(
                expected_sales - wage_bill - indirect_tax - target_profit,
                0.0,
            )
            startup_loan = (wage_bill + fixed_cost) * float(
                cfg.get("startup_cash_months", 2.0)
            )
            startup_loan = min(
                startup_loan,
                12.0
                * state.baseline_total_output
                * float(cfg.get("startup_loan_max_annual_output_share", 0.005)),
            )
            state.bank.firm_credit_requested += startup_loan
            credit_room = min(
                self._bank_available_credit(),
                self._capital_credit_room(
                    float(self.config["banking"].get("firm_loan_risk_weight", 1.0))
                ),
            )
            if startup_loan <= 0.0 or credit_room + 1e-9 < startup_loan:
                state.bank.firm_credit_rejected += startup_loan
                break

            firm_id = state.next_firm_id
            state.next_firm_id += 1
            for resident in recruits:
                resident.employed = True
                resident.shock_unemployed = False
                resident.unemployment_duration = 0
                resident.shock_unemployment_duration = 0
                resident.monthly_work_hours = float(
                    self.config["simulation"]["monthly_work_hours"]
                )
                resident.employer_id = firm_id
                resident.gross_wage = resident.baseline_gross_wage * wage_factor
            state.firms[firm_id] = Firm(
                id=firm_id,
                firm_type=str(type_spec["name"]),
                size_tier="startup",
                employee_ids=[resident.id for resident in recruits],
                ai_target=ai_target,
                labor_displacement=0.0,
                culture=culture,
                ai_multiplier=ai_multiplier,
                expected_demand=expected_sales,
                capacity=capacity,
                sales=0.0,
                price=price,
                wage_bill=wage_bill,
                fixed_cost=fixed_cost,
                cash=startup_loan,
                initial_cash=startup_loan,
                market_share=0.0,
                baseline_quantity=expected_sales / max(price, 1e-9),
                baseline_price=price,
                bank_debt=startup_loan,
                founding_month=month,
                initial_employee_count=len(recruits),
                peak_employee_count=len(recruits),
            )
            state.bank.reserves -= startup_loan
            state.bank.firm_loans += startup_loan
            state.bank.firm_credit_disbursed += startup_loan
            unemployed = unemployed[jobs_per_firm:]
            entries += 1
            jobs_created += len(recruits)
        if entries:
            state.cumulative_firm_entries += entries
            state.cumulative_entry_jobs += jobs_created
            self._update_loan_provisions()
        state.bank.firm_credit_rejected = max(
            state.bank.firm_credit_rejected,
            state.bank.firm_credit_requested - state.bank.firm_credit_disbursed,
        )
        return entries, jobs_created

    def _adjust_employment(self) -> tuple[int, int]:
        cfg = self.config["firms"]
        state = self.state
        fired = 0
        hired = 0
        responsibility_active = self._institution_is_active(
            "employment_responsibility"
        )
        responsibility_cfg = self._institutional_config(
            "employment_responsibility"
        )
        standard_hours = float(self.config["simulation"]["monthly_work_hours"])
        available = sorted((r for r in state.residents.values() if not r.employed), key=lambda r: r.base_productivity, reverse=True)
        for firm in state.firms.values():
            firm.ai_attributable_layoffs_blocked = 0
            firm.distress_exemption_layoffs = 0
            speed = float(cfg["demand_expectation_speed"])
            firm.expected_demand += speed * (firm.sales - firm.expected_demand)
            current = len(firm.employee_ids)
            employees = [state.residents[rid] for rid in firm.employee_ids]
            avg_productivity = (
                mean([self._effective_productivity(r) for r in employees])
                if employees
                else 1.0
            )
            desired = firm.expected_demand / max(float(cfg["target_utilization"]) * state.productivity_scale * firm.ai_multiplier * avg_productivity * firm.price, 1.0)
            shadow_avg_productivity = (
                mean([resident.base_productivity for resident in employees])
                if employees
                else 1.0
            )
            shadow_desired = firm.expected_demand / max(
                float(cfg["target_utilization"])
                * state.productivity_scale
                * shadow_avg_productivity
                * firm.price,
                1.0,
            )
            firm.ai_labor_demand = desired
            firm.shadow_no_ai_labor_demand = shadow_desired
            culture = self._culture_spec(firm)
            workforce_anchor = max(firm.initial_employee_count, 1)
            target_ai_gain = max(firm.ai_target - 1.0, 1e-9)
            ai_progress = min(max((firm.ai_multiplier - 1.0) / target_ai_gain, 0.0), 1.0)
            complementary_jobs = round(
                workforce_anchor
                * float(culture.get("ai_complementary_job_share", 0.0))
                * ai_progress
            )
            target = max(1, round(desired) + complementary_jobs)
            retention_floor = math.ceil(
                workforce_anchor
                * float(culture.get("retention_commitment", 0.0))
            )
            target = max(target, retention_floor)
            demand_outlook = firm.expected_demand / max(
                firm.baseline_quantity * firm.baseline_price, 1.0
            )
            if (
                target < current
                and demand_outlook
                >= float(culture.get("layoff_demand_threshold", 0.0))
                and firm.loss_months
                < int(culture.get("layoff_delay_months", 0))
            ):
                target = current
            adjustment_speed = float(cfg["labor_adjustment_speed"]) * float(
                culture.get("labor_adjustment_multiplier", 1.0)
            )
            if firm.labor_stance == "patient":
                adjustment_speed *= 0.60
            elif firm.labor_stance == "aggressive":
                adjustment_speed *= 1.40
            no_policy_target = target
            exempt = bool(
                responsibility_cfg.get("allow_distress_exemption", True)
                and max(firm.distressed_months, firm.loss_months)
                >= int(
                    responsibility_cfg.get("distress_exemption_months", 6)
                )
            )
            if responsibility_active and not exempt:
                protected_share = float(
                    responsibility_cfg.get(
                        "protected_shadow_employment_share", 1.0
                    )
                )
                firm.protected_job_floor = max(
                    1, round(protected_share * shadow_desired)
                )
                target = max(target, firm.protected_job_floor)
            else:
                firm.protected_job_floor = 0
            if firm.distressed_months >= int(cfg["cash_distress_months"]):
                if not responsibility_active or exempt:
                    target = min(
                        target,
                        math.floor(
                            current
                            * (1.0 - float(cfg["distress_labor_cut"]))
                        ),
                    )
                no_policy_target = min(
                    no_policy_target,
                    math.floor(
                        current * (1.0 - float(cfg["distress_labor_cut"]))
                    ),
                )
            no_policy_change = (
                max(
                    1,
                    math.ceil(
                        abs(no_policy_target - current) * adjustment_speed
                    ),
                )
                if no_policy_target != current
                else 0
            )
            no_policy_fires = (
                min(
                    current - no_policy_target,
                    no_policy_change,
                    max(current - 1, 0),
                )
                if no_policy_target < current
                else 0
            )
            max_change = max(1, math.ceil(abs(target - current) * adjustment_speed)) if target != current else 0
            actual_fires = 0
            if target < current:
                count = min(current - target, max_change, max(current - 1, 0))
                victims = sorted(employees, key=lambda r: (r.base_productivity, -r.gross_wage))[:count]
                for resident in victims:
                    firm.employee_ids.remove(resident.id)
                    resident.employed = False
                    resident.shock_unemployed = True
                    resident.shock_unemployment_duration = 0
                    resident.monthly_work_hours = 0.0
                    resident.employer_id = None
                    fired += 1
                    actual_fires += 1
                    available.append(resident)
            elif target > current and available:
                count = min(target - current, max_change, len(available))
                for _ in range(count):
                    resident = available.pop(0)
                    resident.employed = True
                    resident.shock_unemployed = False
                    resident.unemployment_duration = 0
                    resident.shock_unemployment_duration = 0
                    resident.monthly_work_hours = float(self.config["simulation"]["monthly_work_hours"])
                    resident.employer_id = firm.id
                    resident.gross_wage = resident.baseline_gross_wage * (1.0 + float(cfg["wage_productivity_pass_through"]) * (firm.ai_multiplier - 1.0))
                    firm.employee_ids.append(resident.id)
                    hired += 1
            blocked = max(no_policy_fires - actual_fires, 0)
            if responsibility_active and exempt:
                firm.distress_exemption_layoffs = actual_fires
                state.cumulative_distress_exemption_layoffs += actual_fires
            firm.ai_attributable_layoffs_blocked = blocked
            state.cumulative_ai_attributable_layoffs_blocked += blocked
            final_headcount = len(firm.employee_ids)
            firm.work_intensity = min(
                max(desired / max(final_headcount, 1), 0.0),
                1.0,
            )
            required_hours = max(
                float(
                    responsibility_cfg.get("minimum_required_hours", 0.0)
                ),
                standard_hours * firm.work_intensity,
            )
            for resident_id in firm.employee_ids:
                state.residents[resident_id].monthly_work_hours = (
                    required_hours if responsibility_active else standard_hours
                )
        return hired, fired

    def _normalized_market_shares(self) -> list[float]:
        firms = list(self.state.firms.values())
        total = sum(max(firm.market_share, 0.0) for firm in firms)
        if total <= 0:
            return [1.0 / len(firms)] * len(firms)
        return [max(firm.market_share, 0.0) / total for firm in firms]

    def _refresh_capacities_and_shares(self) -> None:
        elasticity = float(
            self.config.get("banking", {}).get("capital_output_elasticity", 0.0)
        )
        for firm in self.state.firms.values():
            capital_bonus = 1.0 + elasticity * firm.investment_capital / max(
                firm.baseline_quantity * firm.baseline_price, 1.0
            )
            firm.capacity = (
                self.state.productivity_scale
                * firm.ai_multiplier
                * sum(
                    self._effective_productivity(self.state.residents[rid])
                    for rid in firm.employee_ids
                )
                * capital_bonus
            )
        firms = list(self.state.firms.values())
        total = sum(firm.capacity for firm in firms)
        capacity_shares = [
            firm.capacity / total if total else 1.0 / len(firms) for firm in firms
        ]
        competition = self.config["firms"].get("competition", {})
        if (
            not bool(competition.get("enabled", False))
            or not self._policy_is_active()
            or not bool(self.config["scenario"].get("private_ai", False))
        ):
            for firm, share in zip(firms, capacity_shares):
                firm.market_share = share
            return

        average_price = mean([firm.price for firm in firms]) if firms else 1.0
        strategy = self._government_strategy()
        price_sensitivity = float(
            competition.get("price_sensitivity", 0.0)
        ) * float(strategy.get("price_sensitivity_multiplier", 1.0))
        reputation_weight = float(
            competition.get("employment_reputation_weight", 0.0)
        ) * float(strategy.get("employment_reputation_multiplier", 1.0))
        below_cost_penalty = float(
            strategy.get(
                "below_cost_market_penalty",
                competition.get("below_cost_market_penalty", 1.0),
            )
        )
        attractions: list[float] = []
        for firm, capacity_share in zip(firms, capacity_shares):
            employment_reputation = len(firm.employee_ids) / max(
                firm.initial_employee_count, 1
            )
            attractions.append(
                max(capacity_share, 1e-12)
                ** float(competition.get("capacity_weight", 1.0))
                * (average_price / max(firm.price, 1e-9))
                ** price_sensitivity
                * max(employment_reputation, 0.05)
                ** reputation_weight
            )
            unit_cost = (
                firm.wage_bill + firm.fixed_cost + firm.interest_payment
            ) / max(firm.capacity, 1e-9)
            if firm.price < unit_cost:
                attractions[-1] *= below_cost_penalty
        attraction_total = sum(attractions)
        desired_shares = [
            attraction / attraction_total
            if attraction_total
            else 1.0 / len(firms)
            for attraction in attractions
        ]
        speed = float(competition.get("share_adjustment_speed", 0.25))
        blended = [
            max((1.0 - speed) * firm.market_share + speed * desired, 0.0)
            for firm, desired in zip(firms, desired_shares)
        ]
        blended_total = sum(blended)
        for firm, share in zip(firms, blended):
            firm.market_share = (
                share / blended_total if blended_total else 1.0 / len(firms)
            )
