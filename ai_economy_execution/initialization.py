from __future__ import annotations

import math
import random
from collections.abc import Iterable
from typing import Any

from .models import Bank, EconomyState, Firm, Government, Resident


def _largest_remainder(total: int, shares: Iterable[float]) -> list[int]:
    shares = list(shares)
    raw = [total * value / sum(shares) for value in shares]
    result = [math.floor(value) for value in raw]
    for index in sorted(range(len(raw)), key=lambda i: raw[i] - result[i], reverse=True)[: total - sum(result)]:
        result[index] += 1
    return result


def _allocate_slots(total: int, weights: list[float]) -> list[int]:
    if total < len(weights):
        raise ValueError("There must be at least one employed resident per firm")
    base = [1] * len(weights)
    extra = _largest_remainder(total - len(weights), weights)
    return [one + more for one, more in zip(base, extra)]


def _rebalance_assignments(
    assignments: list[list[int]],
    residents: dict[int, Resident],
    max_wage_per_productivity: float,
) -> None:
    """Swap workers until each firm's capacity-based sales share can cover wages."""
    wages = [sum(residents[rid].gross_wage for rid in ids) for ids in assignments]
    productivity = [sum(residents[rid].base_productivity for rid in ids) for ids in assignments]

    def ratio(index: int) -> float:
        return wages[index] / max(productivity[index], 1e-12)

    def violation(index: int) -> float:
        return max(ratio(index) - max_wage_per_productivity, 0.0)

    for _ in range(2000):
        worst = max(range(len(assignments)), key=violation)
        if violation(worst) <= 1e-9:
            return
        best: tuple[float, float, int, int, int] | None = None
        old_worst_violation = violation(worst)
        for other in range(len(assignments)):
            if other == worst:
                continue
            old_score = old_worst_violation**2 + violation(other) ** 2
            for worst_position, worst_id in enumerate(assignments[worst]):
                worst_resident = residents[worst_id]
                for other_position, other_id in enumerate(assignments[other]):
                    other_resident = residents[other_id]
                    new_worst_wage = wages[worst] - worst_resident.gross_wage + other_resident.gross_wage
                    new_other_wage = wages[other] - other_resident.gross_wage + worst_resident.gross_wage
                    new_worst_productivity = (
                        productivity[worst] - worst_resident.base_productivity + other_resident.base_productivity
                    )
                    new_other_productivity = (
                        productivity[other] - other_resident.base_productivity + worst_resident.base_productivity
                    )
                    new_worst_ratio = new_worst_wage / max(new_worst_productivity, 1e-12)
                    new_other_ratio = new_other_wage / max(new_other_productivity, 1e-12)
                    new_worst_violation = max(new_worst_ratio - max_wage_per_productivity, 0.0)
                    new_other_violation = max(new_other_ratio - max_wage_per_productivity, 0.0)
                    new_score = new_worst_violation**2 + new_other_violation**2
                    if new_score >= old_score - 1e-12:
                        continue
                    candidate = (
                        new_score,
                        max(new_worst_ratio, new_other_ratio),
                        other,
                        worst_position,
                        other_position,
                    )
                    if best is None or candidate < best:
                        best = candidate
        if best is None:
            break
        _, _, other, worst_position, other_position = best
        worst_id = assignments[worst][worst_position]
        other_id = assignments[other][other_position]
        worst_resident = residents[worst_id]
        other_resident = residents[other_id]
        assignments[worst][worst_position], assignments[other][other_position] = other_id, worst_id
        wages[worst] += other_resident.gross_wage - worst_resident.gross_wage
        wages[other] += worst_resident.gross_wage - other_resident.gross_wage
        productivity[worst] += other_resident.base_productivity - worst_resident.base_productivity
        productivity[other] += worst_resident.base_productivity - other_resident.base_productivity

    worst = max(range(len(assignments)), key=ratio)
    raise ValueError(
        "Unable to calibrate non-negative firm fixed costs after worker rebalancing; "
        f"worst wage/productivity ratio={ratio(worst):.6f}, allowed={max_wage_per_productivity:.6f}"
    )


def _make_baseline_consumption_feasible(
    residents: dict[int, Resident], minimum_consumption_ratio: float
) -> None:
    """Keep each income group's consumption total while removing forced cash drawdown at equilibrium."""
    groups = sorted({resident.income_group for resident in residents.values()})
    for group in groups:
        selected = [resident for resident in residents.values() if resident.income_group == group]
        cash_months = {
            resident.id: resident.initial_cash / max(resident.baseline_consumption, 1e-12)
            for resident in selected
        }
        group_target = sum(resident.baseline_consumption for resident in selected)
        for resident in selected:
            resident.baseline_consumption = min(
                resident.baseline_consumption, resident.baseline_disposable_income
            )
        remainder = group_target - sum(resident.baseline_consumption for resident in selected)
        while remainder > 1e-9:
            receivers = [
                resident
                for resident in selected
                if resident.baseline_disposable_income - resident.baseline_consumption > 1e-9
            ]
            room = sum(
                resident.baseline_disposable_income - resident.baseline_consumption
                for resident in receivers
            )
            if not receivers or room + 1e-9 < remainder:
                raise ValueError(
                    f"Income group {group!r} cannot finance its calibrated baseline consumption"
                )
            for resident in receivers:
                free = resident.baseline_disposable_income - resident.baseline_consumption
                addition = min(free, remainder * free / room)
                resident.baseline_consumption += addition
            remainder = group_target - sum(resident.baseline_consumption for resident in selected)

        for resident in selected:
            resident.minimum_consumption = (
                resident.baseline_consumption * minimum_consumption_ratio
            )
            resident.cash = resident.baseline_consumption * cash_months[resident.id]
            resident.initial_cash = resident.cash
            resident.target_cash = resident.cash
            resident.disposable_income = resident.baseline_disposable_income
            resident.nominal_consumption = resident.baseline_consumption
            resident.real_consumption = resident.baseline_consumption


def initialize_economy(config: dict[str, Any]) -> EconomyState:
    sim = config["simulation"]
    household_cfg = config["households"]
    firm_cfg = config["firms"]
    gov_cfg = config["government"]
    scenario = config.get("active_scenario", "E0")
    rng = random.Random(int(sim["seed"]))

    population = int(sim["population"])
    group_specs = household_cfg["income_groups"]
    group_counts = _largest_remainder(population, [g["share"] for g in group_specs])
    unemployed_total = round(population * float(sim["initial_unemployment_rate"]))
    unemployed_counts = _largest_remainder(unemployed_total, group_counts)
    tax_rate = float(household_cfg["labor_tax_rate"])

    displayed_consumption_total = sum(
        count * float(group["baseline_consumption"]) for group, count in zip(group_specs, group_counts)
    )
    target_consumption_total = population * float(household_cfg["annual_consumption_per_household"]) / 12.0
    consumption_scale = target_consumption_total / displayed_consumption_total

    residents: dict[int, Resident] = {}
    next_id = 1
    for group, count, unemployed_count in zip(group_specs, group_counts, unemployed_counts):
        employment_rate = (count - unemployed_count) / count
        group_net_wage = float(group["monthly_disposable_income"]) * float(group["net_wage_share"])
        employed_gross_wage = group_net_wage / employment_rate / (1.0 - tax_rate)
        other_income = float(group["monthly_disposable_income"]) - group_net_wage
        statuses = [False] * unemployed_count + [True] * (count - unemployed_count)
        rng.shuffle(statuses)
        productivity_draws = [rng.lognormvariate(-0.5 * 0.12**2, 0.12) for _ in range(count)]
        mean_productivity = sum(productivity_draws) / count
        for employed, productivity in zip(statuses, productivity_draws):
            baseline_consumption = float(group["baseline_consumption"]) * consumption_scale
            cash = baseline_consumption * float(group["cash_months"])
            wage = employed_gross_wage if employed else 0.0
            residents[next_id] = Resident(
                id=next_id,
                income_group=str(group["name"]),
                base_productivity=productivity / mean_productivity,
                employed=employed,
                employer_id=None,
                gross_wage=wage,
                baseline_gross_wage=employed_gross_wage,
                other_baseline_income=other_income,
                monthly_disposable_anchor=float(group["monthly_disposable_income"]),
                baseline_consumption=baseline_consumption,
                consumption_propensity=float(group["consumption_propensity"]),
                minimum_consumption=baseline_consumption * float(household_cfg["minimum_consumption_ratio"]),
                cash=cash,
                initial_cash=cash,
                target_cash=cash,
                monthly_work_hours=float(sim["monthly_work_hours"]) if employed else 0.0,
                last_net_wage=employed_gross_wage * (1.0 - tax_rate),
            )
            next_id += 1

    employed = [resident for resident in residents.values() if resident.employed]
    employed_productivity_mean = sum(resident.base_productivity for resident in employed) / len(employed)
    for resident in employed:
        resident.base_productivity /= employed_productivity_mean
    firm_count = sim.get("firm_count")
    if firm_count is None:
        firm_count = round(len(employed) / float(sim["average_firm_size"]))
    firm_count = max(1, min(int(firm_count), len(employed)))

    tier_specs = firm_cfg["size_tiers"]
    tier_counts = _largest_remainder(firm_count, [tier["share"] for tier in tier_specs])
    tier_names: list[str] = []
    tier_weights: list[float] = []
    for tier, count in zip(tier_specs, tier_counts):
        tier_names.extend([str(tier["name"])] * count)
        tier_weights.extend([float(tier["weight"])] * count)
    paired = list(zip(tier_names, tier_weights))
    rng.shuffle(paired)
    tier_names = [name for name, _ in paired]
    tier_weights = [weight for _, weight in paired]
    slots = _allocate_slots(len(employed), tier_weights)

    # First balance wage totals per slot, then repair rare productivity-cost
    # mismatches with pairwise worker swaps while preserving firm sizes.
    resident_lookup = {resident.id: resident for resident in employed}
    assignments = [[] for _ in slots]
    remaining = slots[:]
    wage_totals = [0.0] * firm_count
    for resident in sorted(employed, key=lambda r: r.gross_wage, reverse=True):
        eligible = [i for i, value in enumerate(remaining) if value]
        index = min(eligible, key=lambda i: (wage_totals[i] / slots[i], len(assignments[i]) / slots[i]))
        assignments[index].append(resident.id)
        wage_totals[index] += resident.gross_wage
        remaining[index] -= 1

    baseline_transfers = sum(r.baseline_consumption for r in residents.values()) * float(
        gov_cfg["base_transfer_share"]
    )
    unemployed = [r for r in residents.values() if not r.employed]
    if unemployed:
        weights = [
            max(
                float(gov_cfg["minimum_unemployment_transfer"]),
                r.baseline_gross_wage
                * (1.0 - tax_rate)
                * float(gov_cfg["unemployment_replacement_rate"]),
            )
            for r in unemployed
        ]
        allocations = [baseline_transfers * weight / sum(weights) for weight in weights]
        for _ in range(len(unemployed)):
            overflow = sum(
                max(value - resident.other_baseline_income, 0.0)
                for resident, value in zip(unemployed, allocations)
            )
            allocations = [
                min(value, resident.other_baseline_income)
                for resident, value in zip(unemployed, allocations)
            ]
            if overflow <= 1e-9:
                break
            room = [
                max(resident.other_baseline_income - value, 0.0)
                for resident, value in zip(unemployed, allocations)
            ]
            if sum(room) <= 0:
                break
            allocations = [
                value + overflow * free / sum(room)
                for value, free in zip(allocations, room)
            ]
        for resident, transfer in zip(unemployed, allocations):
            resident.redistributed_income = transfer
            resident.baseline_transfer = transfer
            resident.other_baseline_income -= transfer

    for resident in residents.values():
        resident.baseline_disposable_income = (
            (resident.gross_wage * (1.0 - tax_rate) if resident.employed else 0.0)
            + resident.other_baseline_income
            + resident.baseline_transfer
        )
    _make_baseline_consumption_feasible(
        residents, float(household_cfg["minimum_consumption_ratio"])
    )

    baseline_household_demand = sum(r.baseline_consumption for r in residents.values())
    public_service = baseline_household_demand * float(gov_cfg["public_service_share"])
    ai_spending = (
        public_service
        * float(gov_cfg["ai_budget_share_of_service"])
        * float(gov_cfg["ai_use_rate"])
    )
    gross_wages = sum(r.gross_wage for r in residents.values())
    labor_tax = gross_wages * tax_rate
    transfers = sum(r.baseline_transfer for r in residents.values())
    output_tax_rate = float(gov_cfg["indirect_tax_rate"]) + float(
        firm_cfg["target_pre_tax_margin"]
    ) * float(gov_cfg["firm_profit_tax_rate"])
    balanced_government_purchase = (
        labor_tax + output_tax_rate * baseline_household_demand - transfers
    ) / max(1.0 - output_tax_rate, 1e-12)
    procurement = max(balanced_government_purchase - public_service - ai_spending, 0.0)
    baseline_government_purchase = public_service + procurement + ai_spending
    baseline_total_output = baseline_household_demand + baseline_government_purchase
    wage_capacity_share = 1.0 - float(gov_cfg["indirect_tax_rate"]) - float(firm_cfg["target_pre_tax_margin"])
    max_wage_per_productivity = wage_capacity_share * baseline_total_output / sum(
        resident.base_productivity for resident in employed
    )
    _rebalance_assignments(assignments, resident_lookup, max_wage_per_productivity)

    type_specs = firm_cfg["types"]
    type_counts = _largest_remainder(firm_count, [item["share"] for item in type_specs])
    firm_types: list[dict[str, Any]] = []
    for item, count in zip(type_specs, type_counts):
        firm_types.extend([item] * count)
    rng.shuffle(firm_types)

    culture_specs = list(firm_cfg.get("cultures", [{"name": "adaptive", "share": 1.0}]))
    culture_mode = str(firm_cfg.get("culture_mode", "mixed"))
    if culture_mode == "mixed":
        culture_counts = _largest_remainder(
            firm_count,
            [float(item.get("share", 0.0)) for item in culture_specs],
        )
        firm_cultures: list[str] = []
        for item, count in zip(culture_specs, culture_counts):
            firm_cultures.extend([str(item["name"])] * count)
        rng.shuffle(firm_cultures)
    else:
        valid_cultures = {str(item["name"]) for item in culture_specs}
        if culture_mode not in valid_cultures:
            raise ValueError(
                f"Unknown firm culture {culture_mode!r}; expected 'mixed' or one of {sorted(valid_cultures)}"
            )
        firm_cultures = [culture_mode] * firm_count

    target_utilization = float(firm_cfg["target_utilization"])
    productivity_scale = baseline_total_output / target_utilization / sum(r.base_productivity for r in employed)

    firms: dict[int, Firm] = {}
    capacities: list[float] = []
    for ids in assignments:
        capacities.append(productivity_scale * sum(resident_lookup[rid].base_productivity for rid in ids))
    total_capacity = sum(capacities)
    for index, (ids, type_spec, culture, tier_name, capacity) in enumerate(
        zip(assignments, firm_types, firm_cultures, tier_names, capacities), start=1
    ):
        firm_id = 10000 + index
        for resident_id in ids:
            residents[resident_id].employer_id = firm_id
        share = capacity / total_capacity
        sales = baseline_total_output * share
        wage_bill = sum(residents[rid].gross_wage for rid in ids)
        indirect_tax = sales * float(gov_cfg["indirect_tax_rate"])
        target_profit = sales * float(firm_cfg["target_pre_tax_margin"])
        fixed_cost = sales - wage_bill - indirect_tax - target_profit
        if fixed_cost < -1e-6:
            raise ValueError(f"Firm {firm_id} has negative calibrated fixed cost ({fixed_cost:.2f})")
        fixed_cost = max(0.0, fixed_cost)
        initial_cash = (wage_bill + fixed_cost) * float(firm_cfg["cash_months"])
        firms[firm_id] = Firm(
            id=firm_id,
            firm_type=str(type_spec["name"]),
            size_tier=tier_name,
            employee_ids=ids,
            ai_target=float(type_spec["ai_target"]),
            # Retained on Firm only so older checkpoints remain readable. It is
            # deliberately zero and is not used by the employment equation.
            labor_displacement=0.0,
            culture=culture,
            expected_demand=sales,
            capacity=capacity,
            sales=sales,
            wage_bill=wage_bill,
            fixed_cost=fixed_cost,
            cash=initial_cash,
            initial_cash=initial_cash,
            market_share=share,
            baseline_quantity=sales,
            baseline_price=1.0,
            founding_month=0,
            initial_employee_count=len(ids),
            peak_employee_count=len(ids),
        )

    sales_tax = baseline_total_output * float(gov_cfg["indirect_tax_rate"])
    profit_tax = baseline_total_output * float(firm_cfg["target_pre_tax_margin"]) * float(gov_cfg["firm_profit_tax_rate"])
    transfers = sum(r.redistributed_income for r in residents.values())
    spending = baseline_government_purchase + transfers
    tax_revenue = labor_tax + sales_tax + profit_tax
    initial_gov_cash = spending * float(gov_cfg["cash_months"])
    government = Government(
        ai_use_rate=float(gov_cfg["ai_use_rate"]),
        cash=initial_gov_cash,
        initial_cash=initial_gov_cash,
        public_service_spending=public_service,
        procurement=procurement,
        ai_spending=ai_spending,
        transfers=transfers,
        tax_revenue=tax_revenue,
        fiscal_balance=tax_revenue - spending,
    )
    banking_cfg = config.get("banking", {})
    initial_bank_equity = (
        12.0
        * baseline_total_output
        * float(banking_cfg.get("initial_equity_to_annual_output", 0.0))
        if banking_cfg.get("enabled", False)
        else 0.0
    )
    bank = Bank(reserves=initial_bank_equity, equity=initial_bank_equity)
    state = EconomyState(
        month=0,
        scenario=scenario,
        seed=int(sim["seed"]),
        residents=residents,
        firms=firms,
        government=government,
        bank=bank,
        baseline_household_demand=baseline_household_demand,
        baseline_government_purchase=baseline_government_purchase,
        baseline_total_output=baseline_total_output,
        productivity_scale=productivity_scale,
        baseline_public_service=public_service,
        baseline_procurement=procurement,
        baseline_ai_spending=ai_spending,
        initial_firm_count=len(firms),
        next_firm_id=max(firms, default=10000) + 1,
        price_index_basket={
            firm.id: {
                "quantity": firm.baseline_quantity,
                "base_price": firm.baseline_price,
                "last_price": firm.price,
            }
            for firm in firms.values()
        },
        initial_culture_employment={
            culture: sum(
                len(firm.employee_ids)
                for firm in firms.values()
                if firm.culture == culture
            )
            for culture in {firm.culture for firm in firms.values()}
        },
    )
    validate_initial_state(state)
    return state


def validate_initial_state(state: EconomyState) -> None:
    employed_ids = {r.id for r in state.residents.values() if r.employed}
    assigned = [rid for firm in state.firms.values() for rid in firm.employee_ids]
    if len(assigned) != len(set(assigned)) or set(assigned) != employed_ids:
        raise ValueError("Employee-firm assignment is not one-to-one")
    wage_from_firms = sum(firm.wage_bill for firm in state.firms.values())
    wage_from_residents = sum(r.gross_wage for r in state.residents.values())
    if not math.isclose(wage_from_firms, wage_from_residents, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Firm and resident gross wages do not reconcile")
    if not math.isclose(
        sum(firm.sales for firm in state.firms.values()), state.baseline_total_output, rel_tol=0, abs_tol=1e-6
    ):
        raise ValueError("Initial firm sales do not reconcile to final demand")


def build_agent_specs(state: EconomyState, llm_roles: set[str] | None = None) -> list[dict[str, Any]]:
    llm_roles = llm_roles or set()
    specs: list[dict[str, Any]] = []
    for resident in state.residents.values():
        specs.append({"id": resident.id, "profile": {"id": resident.id, "name": f"Resident-{resident.id}", "role": "resident", "economic_id": resident.id}, "config": {"llm_enabled": "resident" in llm_roles}})
    for firm in state.firms.values():
        specs.append({"id": firm.id, "profile": {"id": firm.id, "name": f"Firm-{firm.id}", "role": "firm", "economic_id": firm.id}, "config": {"llm_enabled": "firm" in llm_roles}})
    government = state.government
    specs.append({"id": government.id, "profile": {"id": government.id, "name": "Government", "role": "government", "economic_id": government.id}, "config": {"llm_enabled": "government" in llm_roles}})
    return specs
