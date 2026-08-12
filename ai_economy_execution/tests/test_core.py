from __future__ import annotations

import copy
import unittest
from pathlib import Path
from statistics import mean

from ai_economy_execution.configuration import load_config, scenario_config
from ai_economy_execution.core import EconomyEngine, bounded_intent, fixed_basket_price_index
from ai_economy_execution.initialization import initialize_economy
from ai_economy_execution.gates import audit_history, gate_thresholds
from ai_economy_execution.metrics import (
    resident_distribution_metrics,
    summarize,
    validate_metric,
)
from ai_economy_execution.models import Firm
from ai_economy_execution.result_layout import (
    matrix_aggregate_dir,
    matrix_cell_dir,
    resolve_cognitive_regime,
)


class InitializationTests(unittest.TestCase):
    def test_baseline_scale_and_accounting(self) -> None:
        state = initialize_economy(scenario_config(load_config(), "E0", 500, 1))
        self.assertEqual(len(state.residents), 500)
        self.assertEqual(len(state.firms), 30)
        self.assertEqual(sum(r.employed for r in state.residents.values()), 474)
        self.assertAlmostEqual(state.baseline_household_demand, 1_228_166.6666667, places=5)
        self.assertAlmostEqual(state.baseline_government_purchase, 109_585.3676745, places=5)
        self.assertAlmostEqual(state.productivity_scale, state.baseline_total_output / 0.85 / 474, places=8)
        self.assertAlmostEqual(state.government.transfers, state.baseline_household_demand * 0.02, places=5)
        self.assertAlmostEqual(state.government.fiscal_balance, 0.0, places=6)
        self.assertTrue(all(
            resident.baseline_consumption <= resident.baseline_disposable_income + 1e-9
            for resident in state.residents.values()
        ))
        assigned = [rid for firm in state.firms.values() for rid in firm.employee_ids]
        self.assertEqual(len(assigned), len(set(assigned)))

    def test_homogeneous_culture_assignment(self) -> None:
        config = scenario_config(load_config(), "E1", 100, 1)
        config["firms"]["culture_mode"] = "augmentation"
        state = initialize_economy(config)
        self.assertEqual({firm.culture for firm in state.firms.values()}, {"augmentation"})

    def test_initial_observations_describe_equilibrium_gaps(self) -> None:
        config = scenario_config(load_config(), "E0", 500, 1)
        state = initialize_economy(config)
        engine = EconomyEngine(state, config)
        resident = engine.observe(next(iter(state.residents)))
        firm = engine.observe(next(iter(state.firms)))
        government = engine.observe(state.government.id)
        self.assertAlmostEqual(resident["income_gap_ratio"], 0.0)
        self.assertAlmostEqual(resident["cash_gap_months"], 0.0)
        self.assertGreater(resident["last_disposable_income"], 0.0)
        self.assertAlmostEqual(firm["utilization_gap"], 0.0)
        self.assertAlmostEqual(government["unemployment_gap"], 0.0)
        self.assertFalse(resident["shock_active"])
        for observation in (resident, firm, government):
            self.assertFalse(observation["trend_available"])
            self.assertEqual(observation["trend_window_months"], 3)
        self.assertEqual(resident["unemployment_change_3m"], 0.0)
        self.assertEqual(firm["firm_sales_change_3m"], 0.0)
        self.assertEqual(government["debt_ratio_change_3m"], 0.0)

    def test_population_scaling_preserves_density(self) -> None:
        states = [initialize_economy(scenario_config(load_config(), "E0", n, 1)) for n in (500, 1000)]
        self.assertEqual([len(state.firms) for state in states], [30, 60])
        self.assertAlmostEqual(states[0].baseline_total_output / 500, states[1].baseline_total_output / 1000)
        self.assertAlmostEqual(states[0].productivity_scale, states[1].productivity_scale, places=6)

    def test_older_checkpoint_defaults_missing_shock_duration(self) -> None:
        from ai_economy_execution.models import EconomyState

        state = initialize_economy(scenario_config(load_config(), "E0", 100, 1))
        raw = state.to_dict()
        for resident in raw["residents"].values():
            resident.pop("shock_unemployment_duration", None)

        restored = EconomyState.from_dict(raw)

        self.assertTrue(
            all(
                resident.shock_unemployment_duration == 0
                for resident in restored.residents.values()
            )
        )


class EngineTests(unittest.TestCase):
    def test_observation_trends_are_deterministic_and_read_only(self) -> None:
        config = scenario_config(load_config(), "E0", 100, 1)
        state = initialize_economy(config)
        state.history = [
            {
                "unemployment_rate": 0.05,
                "aggregate_price": 1.00,
                "real_consumption": 100.0,
                "firm_sales": 100.0,
                "capacity_utilization": 0.80,
                "market_hhi": 0.25,
                "household_consumption": 100.0,
                "government_debt_ratio": 0.10,
            },
            {
                "unemployment_rate": 0.06,
                "aggregate_price": 1.05,
                "real_consumption": 98.0,
                "firm_sales": 105.0,
                "capacity_utilization": 0.85,
                "market_hhi": 0.23,
                "household_consumption": 98.0,
                "government_debt_ratio": 0.11,
            },
            {
                "unemployment_rate": 0.03,
                "aggregate_price": 1.10,
                "real_consumption": 95.0,
                "firm_sales": 110.0,
                "capacity_utilization": 0.90,
                "market_hhi": 0.20,
                "household_consumption": 96.0,
                "government_debt_ratio": 0.12,
            },
        ]
        history_before = copy.deepcopy(state.history)
        engine = EconomyEngine(state, config)

        resident = engine.observe(next(iter(state.residents)))
        firm = engine.observe(next(iter(state.firms)))
        government = engine.observe(state.government.id)

        self.assertTrue(resident["trend_available"])
        self.assertAlmostEqual(resident["unemployment_change_3m"], -0.02)
        self.assertAlmostEqual(resident["aggregate_price_change_3m"], 0.10)
        self.assertAlmostEqual(resident["real_consumption_change_3m"], -0.05)
        self.assertAlmostEqual(firm["firm_sales_change_3m"], 0.10)
        self.assertAlmostEqual(firm["capacity_utilization_change_3m"], 0.10)
        self.assertAlmostEqual(firm["market_hhi_change_3m"], -0.05)
        self.assertAlmostEqual(
            government["household_consumption_change_3m"], -0.04
        )
        self.assertAlmostEqual(government["debt_ratio_change_3m"], 0.02)
        self.assertEqual(state.history, history_before)

    def test_cash_target_rises_with_risk_and_excess_cash_is_only_partly_swept(self) -> None:
        config = scenario_config(load_config(), "E1", 100, 1)
        state = initialize_economy(config)
        resident = next(iter(state.residents.values()))
        resident.employed = False
        resident.shock_unemployed = True
        resident.unemployment_duration = int(
            config["households"]["unemployment_stress_months"]
        )
        resident.shock_unemployment_duration = int(
            config["households"]["unemployment_stress_months"]
        )
        resident.disposable_income = 0.0
        resident.cash = 4.0 * resident.initial_cash
        engine = EconomyEngine(state, config)

        deposits, managed_funds, _ = engine._sweep_household_deposits()

        self.assertAlmostEqual(resident.target_cash, 3.0 * resident.initial_cash)
        self.assertGreater(resident.cash, resident.target_cash)
        self.assertGreater(deposits + managed_funds, 0.0)
        self.assertNotAlmostEqual(resident.cash, resident.initial_cash)

    def test_asset_balance_does_not_mask_consumption_and_income_stress(self) -> None:
        config = scenario_config(load_config(), "E1", 100, 1)
        resident = next(iter(initialize_economy(config).residents.values()))
        resident.cash = 10.0 * resident.minimum_consumption
        resident.deposits = 10.0 * resident.minimum_consumption
        resident.disposable_income = 0.0
        resident.real_consumption = 0.0
        resident.employed = False
        resident.shock_unemployed = True
        resident.unemployment_duration = 12
        resident.shock_unemployment_duration = 12

        metrics = resident_distribution_metrics([resident])

        self.assertEqual(metrics["liquidity_vulnerable_rate"], 0.0)
        self.assertEqual(metrics["consumption_compression_rate"], 1.0)
        self.assertEqual(metrics["income_stress_rate"], 1.0)
        self.assertEqual(metrics["economic_stress_rate"], 1.0)

    def test_general_and_shock_unemployment_spells_are_tracked_separately(self) -> None:
        config = scenario_config(load_config(), "E0", 100, 1)
        state = initialize_economy(config)
        structural = next(
            resident for resident in state.residents.values() if not resident.employed
        )
        displaced = next(
            resident for resident in state.residents.values() if resident.employed
        )
        displaced.employed = False
        displaced.shock_unemployed = True
        displaced.employer_id = None
        displaced.gross_wage = 0.0
        for firm in state.firms.values():
            if displaced.id in firm.employee_ids:
                firm.employee_ids.remove(displaced.id)
                break
        engine = EconomyEngine(state, config)

        engine.step()

        self.assertEqual(structural.unemployment_duration, 1)
        self.assertEqual(structural.shock_unemployment_duration, 0)
        self.assertEqual(displaced.unemployment_duration, 1)
        self.assertEqual(displaced.shock_unemployment_duration, 1)

        structural.unemployment_duration = int(
            config["households"]["unemployment_stress_months"]
        )
        metrics = resident_distribution_metrics([structural])
        self.assertEqual(metrics["persistent_unemployment_rate"], 1.0)
        self.assertEqual(metrics["shock_persistent_unemployment_rate"], 0.0)

    def test_failed_firm_exits_and_bank_recognizes_loss(self) -> None:
        config = scenario_config(load_config(), "E1", 100, 1)
        config["firms"]["bankruptcy_minimum_age_months"] = 0
        config["firms"]["bankruptcy_cash_distress_months"] = 1
        config["firms"]["max_monthly_exits"] = 1
        state = initialize_economy(config)
        firm = next(iter(state.firms.values()))
        employee_ids = list(firm.employee_ids)
        exposure = 1_000.0
        firm.cash = -1.0
        firm.distressed_months = 1
        firm.bank_debt = exposure
        state.bank.reserves -= exposure
        state.bank.firm_loans += exposure
        engine = EconomyEngine(state, config)

        exits, jobs_lost = engine._process_firm_exits(
            int(config["simulation"]["shock_month"])
        )

        self.assertEqual(exits, 1)
        self.assertEqual(jobs_lost, len(employee_ids))
        self.assertNotIn(firm.id, state.firms)
        self.assertTrue(all(not state.residents[rid].employed for rid in employee_ids))
        self.assertAlmostEqual(state.bank.firm_loans, 0.0)
        self.assertAlmostEqual(state.bank.writeoffs, exposure)
        self.assertAlmostEqual(state.bank.balance_sheet_error(), 0.0, delta=1e-6)

    def test_ai_opportunity_finances_new_firm_and_jobs(self) -> None:
        config = scenario_config(load_config(), "E1", 100, 1)
        config["banking"]["activation_month"] = 1
        config["firms"]["entry_interval_months"] = 1
        config["firms"]["entry_unemployment_threshold"] = 0.0
        config["firms"]["entry_ai_opportunity_threshold"] = 0.0
        config["firms"]["entry_unmet_demand_threshold"] = 0.0
        state = initialize_economy(config)
        original_firms = len(state.firms)
        original_employment = sum(r.employed for r in state.residents.values())
        engine = EconomyEngine(state, config)

        entries, jobs_created = engine._process_firm_entries(
            int(config["simulation"]["shock_month"]), 0.0
        )

        self.assertEqual(entries, 1)
        self.assertGreater(jobs_created, 0)
        self.assertEqual(len(state.firms), original_firms + 1)
        self.assertEqual(
            sum(r.employed for r in state.residents.values()),
            original_employment + jobs_created,
        )
        entrant = state.firms[max(state.firms)]
        self.assertEqual(entrant.size_tier, "startup")
        self.assertEqual(entrant.ai_multiplier, 1.0)
        self.assertGreater(entrant.bank_debt, 0.0)
        self.assertAlmostEqual(state.bank.balance_sheet_error(), 0.0, delta=1e-6)

    def test_ai_labor_effect_is_only_through_productivity(self) -> None:
        config = scenario_config(load_config(), "E1", 100, 1)
        productivity_only = initialize_economy(config)
        legacy_displacement = copy.deepcopy(productivity_only)

        for firm in productivity_only.firms.values():
            firm.ai_multiplier = firm.ai_target
            firm.labor_displacement = 0.0
        for firm in legacy_displacement.firms.values():
            firm.ai_multiplier = firm.ai_target
            firm.labor_displacement = 1.0

        productivity_result = EconomyEngine(productivity_only, config)._adjust_employment()
        legacy_result = EconomyEngine(legacy_displacement, config)._adjust_employment()

        self.assertGreater(productivity_result[1], 0)
        self.assertEqual(productivity_result, legacy_result)
        self.assertEqual(
            {firm_id: firm.employee_ids for firm_id, firm in productivity_only.firms.items()},
            {firm_id: firm.employee_ids for firm_id, firm in legacy_displacement.firms.items()},
        )

    def test_pre_equilibrium_gate_catches_temporary_departure(self) -> None:
        config = scenario_config(load_config(), "E0", 500, 1)
        template = {
            "employment_rate": 0.948,
            "real_consumption": 100.0,
            "aggregate_price": 1.0,
            "firm_sales": 110.0,
            "sales_identity_error": 0.0,
            "gross_wage_bill": 50.0,
            "wage_identity_error": 0.0,
            "government_tax_revenue": 10.0,
            "tax_identity_error": 0.0,
        }
        history = [{**template, "month": month} for month in range(1, 25)]
        history[4]["employment_rate"] = 0.90
        audit = audit_history(
            history,
            population=500,
            seed=1,
            scenario="E0",
            warmup_months=24,
            thresholds=gate_thresholds(config),
            equilibrium_reference=template,
        )
        self.assertFalse(audit["warmup_stability_pass"])
        self.assertAlmostEqual(audit["warmup_employment_absolute_drift"], 0.048)
        self.assertAlmostEqual(audit["warmup_tail_employment_absolute_drift"], 0.0)

    def test_no_ai_pre_equilibrium_is_flow_stable(self) -> None:
        config = scenario_config(load_config(), "E0", 500, 1)
        state = initialize_economy(config)
        engine = EconomyEngine(state, config)
        for _ in range(int(config["simulation"]["warmup_months"])):
            engine.step()
        first, last = state.history[0], state.history[-1]
        for key in ("employment_rate", "real_consumption", "firm_sales", "aggregate_price"):
            self.assertAlmostEqual(first[key], last[key], places=9)
        self.assertAlmostEqual(last["government_fiscal_balance"], 0.0, places=6)

    def test_policy_scenarios_share_the_complete_pre_shock_path(self) -> None:
        histories = {}
        for scenario in [f"E{i}" for i in range(7)]:
            config = scenario_config(load_config(), scenario, 100, 7)
            state = initialize_economy(config)
            engine = EconomyEngine(state, config)
            for _ in range(int(config["simulation"]["warmup_months"])):
                engine.step()
            histories[scenario] = state.history

        counterfactual_keys = (
            "employment_rate",
            "household_consumption",
            "government_purchase",
            "government_spending",
            "government_debt",
            "public_service_index",
        )
        control = histories["E0"]
        for scenario, history in histories.items():
            for month, (candidate, baseline) in enumerate(zip(history, control), start=1):
                self.assertEqual(
                    {key: candidate[key] for key in counterfactual_keys},
                    {key: baseline[key] for key in counterfactual_keys},
                    f"{scenario} diverged from E0 before treatment at month {month}",
                )

    def test_fixed_basket_price_index_uses_base_quantities(self) -> None:
        firms = [
            Firm(1, "a", "small", [], 1.0, 0.0, price=2.0, baseline_quantity=10.0),
            Firm(2, "b", "small", [], 1.0, 0.0, price=4.0, baseline_quantity=30.0),
        ]
        self.assertAlmostEqual(fixed_basket_price_index(firms), 3.5)

    def test_fixed_price_basket_does_not_add_entrants_or_drop_exits(self) -> None:
        incumbent = Firm(
            1, "a", "small", [], 1.0, 0.0, price=0.8, baseline_quantity=10.0
        )
        basket = {
            1: {"quantity": 10.0, "base_price": 1.0, "last_price": 1.0}
        }
        self.assertAlmostEqual(fixed_basket_price_index([incumbent], basket), 0.8)
        entrant = Firm(
            2, "b", "startup", [], 1.2, 0.0, price=0.5, baseline_quantity=100.0
        )
        self.assertAlmostEqual(fixed_basket_price_index([entrant], basket), 0.8)

    def test_no_ai_control_never_activates_ai_competition(self) -> None:
        histories = []
        for enabled in (False, True):
            config = scenario_config(load_config(), "E0", 100, 3)
            config["firms"]["competition"]["enabled"] = enabled
            state = initialize_economy(config)
            engine = EconomyEngine(state, config)
            for _ in range(60):
                engine.step()
            histories.append(state.history)
        for left, right in zip(*histories):
            for key in (
                "employment_rate",
                "firm_sales",
                "aggregate_price",
                "market_hhi",
            ):
                self.assertAlmostEqual(left[key], right[key], places=9)

    def test_fiscal_limit_preserves_service_floor_and_records_gap(self) -> None:
        config = scenario_config(load_config(), "E4", 100, 9)
        state = initialize_economy(config)
        state.month = int(config["simulation"]["shock_month"]) - 1
        state.government.cash = 0.0
        state.government.tax_revenue = 0.0
        state.government.debt = float(config["government"]["debt_limit_ratio"]) * 12.0 * state.baseline_total_output
        engine = EconomyEngine(state, config)
        base_service = state.baseline_household_demand * float(config["government"]["public_service_share"])
        plan = engine._apply_fiscal_limit({
            "public_service": base_service,
            "procurement": 0.0,
            "ai_spending": 0.0,
            "transfer_extra_pool": 0.0,
        })
        expected_floor = base_service * float(config["government"]["public_service_floor_ratio"])
        self.assertAlmostEqual(plan["public_service"], expected_floor)
        self.assertGreater(state.government.statutory_funding_gap, 0.0)

    def test_household_saving_is_swept_to_deposits_without_destroying_wealth(self) -> None:
        config = scenario_config(load_config(), "E0", 500, 1)
        state = initialize_economy(config)
        initial_cash = sum(resident.cash for resident in state.residents.values())
        engine = EconomyEngine(state, config)
        metric = engine.step()

        self.assertGreater(metric["household_cash"], initial_cash)
        self.assertGreater(metric["household_deposits"], 0.0)
        self.assertAlmostEqual(
            metric["household_financial_wealth"],
            metric["household_cash"]
            + metric["household_deposits"]
            + metric["household_managed_fund_assets"],
            places=6,
        )
        self.assertGreater(metric["household_managed_fund_assets"], 0.0)
        self.assertAlmostEqual(
            metric["bank_managed_funds"],
            metric["household_managed_fund_assets"],
            places=6,
        )
        self.assertAlmostEqual(metric["bank_balance_sheet_error"], 0.0, delta=1e-6)

    def test_bank_funds_government_debt_and_keeps_a_balanced_ledger(self) -> None:
        config = scenario_config(load_config(), "E0", 500, 1)
        config["banking"]["activation_month"] = 2
        state = initialize_economy(config)
        engine = EconomyEngine(state, config)
        engine.step()
        state.government.cash = 0.0
        engine._settle_government(0.0, 1_000.0)

        self.assertAlmostEqual(state.government.debt, 1_000.0)
        self.assertAlmostEqual(state.bank.government_loans, 1_000.0)
        self.assertAlmostEqual(state.bank.balance_sheet_error(), 0.0, delta=1e-6)

    def test_unfunded_government_spending_becomes_arrears(self) -> None:
        config = scenario_config(load_config(), "E6", 100, 1)
        config["banking"]["activation_month"] = 1
        state = initialize_economy(config)
        debt_limit = (
            float(config["government"]["debt_limit_ratio"])
            * 12.0
            * state.baseline_total_output
        )
        state.government.cash = 0.0
        state.government.debt = debt_limit
        state.bank.government_loans = debt_limit
        engine = EconomyEngine(state, config)

        engine._settle_government(0.0, 1_000.0)

        self.assertAlmostEqual(state.government.debt, debt_limit)
        self.assertAlmostEqual(state.government.arrears, 1_000.0)
        self.assertAlmostEqual(state.government.arrears_incurred, 1_000.0)
        self.assertAlmostEqual(state.government.fiscal_shortfall, 1_000.0)
        self.assertAlmostEqual(
            state.government.total_liabilities,
            debt_limit + 1_000.0,
        )
        self.assertAlmostEqual(
            engine.current_macro()["government_debt_ratio"],
            state.government.total_liabilities
            / (12.0 * state.baseline_total_output),
        )

    def test_government_surplus_repays_arrears_before_formal_debt(self) -> None:
        config = scenario_config(load_config(), "E6", 100, 1)
        config["banking"]["activation_month"] = 1
        state = initialize_economy(config)
        state.government.cash = 0.0
        state.government.arrears = 600.0
        state.government.debt = 500.0
        state.bank.government_loans = 500.0
        engine = EconomyEngine(state, config)

        engine._settle_government(1_000.0, 0.0)

        self.assertAlmostEqual(state.government.arrears, 0.0)
        self.assertAlmostEqual(state.government.arrears_repayment, 600.0)
        self.assertAlmostEqual(state.government.debt, 100.0)
        self.assertAlmostEqual(state.bank.government_loans, 100.0)
        self.assertAlmostEqual(state.government.cash, 0.0)

    def test_bank_credit_becomes_firm_investment(self) -> None:
        config = scenario_config(load_config(), "E0", 500, 1)
        config["banking"]["activation_month"] = 2
        state = initialize_economy(config)
        engine = EconomyEngine(state, config)
        engine.step()
        firm = next(iter(state.firms.values()))
        firm.sales = 0.98 * firm.capacity * firm.price

        planned = engine._start_bank_month()
        actual = engine._settle_firm_investment(1.0)

        self.assertGreater(planned, 0.0)
        self.assertAlmostEqual(actual, planned)
        self.assertGreater(firm.bank_debt, 0.0)
        self.assertGreater(firm.investment_capital, 0.0)
        self.assertAlmostEqual(state.bank.balance_sheet_error(), 0.0, delta=1e-6)

    def test_bank_does_not_force_credit_without_qualified_demand(self) -> None:
        config = scenario_config(load_config(), "E0", 500, 1)
        config["banking"]["activation_month"] = 1
        state = initialize_economy(config)
        engine = EconomyEngine(state, config)
        engine.step()
        for firm in state.firms.values():
            firm.sales = 0.98 * firm.capacity * firm.price
            firm.pre_tax_profit = -1.0

        planned = engine._start_bank_month()

        self.assertAlmostEqual(planned, 0.0)
        self.assertAlmostEqual(state.bank.firm_credit_disbursed, 0.0)
        self.assertGreater(state.bank.firm_credit_requested, 0.0)
        self.assertGreater(state.bank.firm_credit_rejected, 0.0)
        self.assertAlmostEqual(state.bank.other_financial_assets, 0.0)

    def test_delinquent_loan_is_provisioned_and_written_off(self) -> None:
        config = scenario_config(load_config(), "E0", 500, 1)
        state = initialize_economy(config)
        engine = EconomyEngine(state, config)
        firm = next(iter(state.firms.values()))
        exposure = 1_000.0
        firm.bank_debt = exposure
        firm.cash = 0.0
        state.bank.firm_loans = exposure
        state.bank.reserves -= exposure

        for _ in range(3):
            engine._service_firm_loans()
        self.assertEqual(firm.loan_status, "substandard")
        self.assertGreater(state.bank.provisions, 0.0)

        for _ in range(int(config["banking"]["writeoff_months"]) - 3):
            engine._service_firm_loans()
        self.assertAlmostEqual(firm.bank_debt, 0.0)
        self.assertAlmostEqual(state.bank.writeoffs, exposure)
        self.assertAlmostEqual(
            state.bank.recoveries,
            exposure * (1.0 - float(config["banking"]["loss_given_default"])),
        )
        self.assertAlmostEqual(state.bank.balance_sheet_error(), 0.0, delta=1e-6)

    def test_banking_metrics_close_over_a_multi_month_policy_path(self) -> None:
        config = scenario_config(load_config(), "E5", 100, 11)
        state = initialize_economy(config)
        engine = EconomyEngine(state, config)
        for _ in range(30):
            metric = engine.step()
            validate_metric(metric)
            self.assertAlmostEqual(
                metric["bank_deposits"],
                metric["household_deposits"],
                places=6,
            )
            self.assertAlmostEqual(
                metric["bank_managed_funds"],
                metric["household_managed_fund_assets"],
                places=6,
            )
            self.assertGreaterEqual(metric["bank_reserve_ratio"], 0.15 - 1e-9)
            self.assertLessEqual(metric["household_saving_to_gdp_ratio"], metric["household_saving_to_output_ratio"])
        self.assertAlmostEqual(
            state.government.debt, state.bank.government_loans, places=6
        )
        self.assertGreater(state.government.public_capital, 0.0)

    def test_all_scenarios_run_and_close_accounts(self) -> None:
        summaries = {}
        for scenario in [f"E{i}" for i in range(7)]:
            config = scenario_config(load_config(), scenario, 100, 11)
            state = initialize_economy(config)
            engine = EconomyEngine(state, config)
            for _ in range(30):
                metric = engine.step()
                validate_metric(metric)
            summaries[scenario] = summarize(state.history)
            self.assertIn("ending_formal_debt_ratio", summaries[scenario])
            self.assertIn("ending_government_arrears_ratio", summaries[scenario])
            self.assertIn("ending_firm_count", summaries[scenario])
            self.assertIn("cumulative_firm_entries", summaries[scenario])
            self.assertIn("cumulative_firm_exits", summaries[scenario])
            self.assertIn("tail_persistent_unemployment_rate", summaries[scenario])
            self.assertIn(
                "tail_shock_persistent_unemployment_rate", summaries[scenario]
            )
            self.assertIn(
                "tail_group_low_shock_persistent_unemployment_rate",
                summaries[scenario],
            )
        self.assertGreater(summaries["E1"]["peak_unemployment_rate"], summaries["E0"]["peak_unemployment_rate"])

    def test_e5_e6_scenario_definitions_are_explicitly_versioned(self) -> None:
        baseline = load_config()
        default_e5 = scenario_config(baseline, "E5", 100, 11)
        default_e6 = scenario_config(baseline, "E6", 100, 11)
        legacy_e5 = scenario_config(
            baseline,
            "E5",
            100,
            11,
            scenario_definition_version="legacy_v1",
        )
        legacy_e6 = scenario_config(
            baseline,
            "E6",
            100,
            11,
            scenario_definition_version="legacy_v1",
        )

        self.assertEqual(
            default_e5["scenario_definition_version"], "institutional_v2"
        )
        self.assertEqual(
            default_e5["scenario"]["name"], "integrated_ai_social_compact"
        )
        for key in (
            "employment_responsibility",
            "ai_infrastructure_levy",
            "solo_enterprise",
            "transfer_response",
            "procurement_response",
            "government_ai",
        ):
            self.assertTrue(default_e5["scenario"][key])
            self.assertTrue(default_e6["scenario"][key])
        self.assertEqual(
            default_e6["scenario"]["name"],
            "integrated_ai_social_compact_fiscal_constraint",
        )
        self.assertEqual(default_e6["scenario"]["max_annual_deficit_ratio"], 0.02)
        self.assertEqual(default_e6["scenario"]["debt_limit_ratio"], 0.40)

        self.assertEqual(legacy_e5["scenario_definition_version"], "legacy_v1")
        self.assertEqual(legacy_e5["scenario"]["name"], "legacy_comprehensive")
        self.assertEqual(
            legacy_e6["scenario"]["name"],
            "legacy_comprehensive_fiscal_constraint",
        )
        for key in (
            "employment_responsibility",
            "ai_infrastructure_levy",
            "solo_enterprise",
        ):
            self.assertFalse(legacy_e5["scenario"].get(key, False))
            self.assertFalse(legacy_e6["scenario"].get(key, False))

        with self.assertRaisesRegex(
            ValueError, "Unknown scenario definition version"
        ):
            scenario_config(
                baseline,
                "E5",
                100,
                11,
                scenario_definition_version="missing",
            )

    def test_culture_changes_price_and_labor_response_without_hard_coding_outcome(self) -> None:
        base = scenario_config(load_config(), "E1", 100, 5)
        base["simulation"]["shock_month"] = 1
        states = {}
        prices = {}
        firings = {}
        for culture in ("augmentation", "cost_cutter"):
            config = copy.deepcopy(base)
            config["firms"]["culture_mode"] = culture
            state = initialize_economy(config)
            engine = EconomyEngine(state, config)
            engine._update_ai(1)
            prices[culture] = sum(firm.price for firm in state.firms.values()) / len(state.firms)
            for firm in state.firms.values():
                firm.expected_demand *= 0.55
            _, firings[culture] = engine._adjust_employment()
            states[culture] = state
        self.assertLess(prices["cost_cutter"], prices["augmentation"])
        self.assertGreater(firings["cost_cutter"], firings["augmentation"])

    def test_active_demand_strategy_expands_real_economy_procurement(self) -> None:
        plans = {}
        for strategy in (
            "passive_safety_net",
            "active_demand",
            "productivity_dividend",
        ):
            config = scenario_config(load_config(), "E5", 100, 7)
            config["simulation"]["shock_month"] = 1
            config["government"]["policy_strategy"] = strategy
            state = initialize_economy(config)
            for firm in state.firms.values():
                firm.ai_multiplier = min(firm.ai_target, 1.10)
            engine = EconomyEngine(state, config)
            plans[strategy] = engine._government_plan(
                {
                    "household_consumption": 0.80 * state.baseline_household_demand,
                    "unemployment_rate": 0.10,
                    "aggregate_price": 0.90,
                }
            )
        self.assertGreater(
            plans["active_demand"]["procurement"],
            plans["passive_safety_net"]["procurement"],
        )
        self.assertGreater(
            plans["active_demand"]["employment_support_procurement"], 0.0
        )
        self.assertGreater(
            plans["productivity_dividend"][
                "productivity_dividend_procurement"
            ],
            0.0,
        )
        self.assertEqual(
            plans["active_demand"]["productivity_dividend_procurement"], 0.0
        )
        self.assertEqual(
            plans["productivity_dividend"]["employment_support_procurement"],
            0.0,
        )

    def test_procurement_scenarios_activate_active_demand_only_after_shock(self) -> None:
        for scenario in ("E5", "E6"):
            config = scenario_config(load_config(), scenario, 100, 7)
            state = initialize_economy(config)
            engine = EconomyEngine(state, config)
            shock_month = int(config["simulation"]["shock_month"])

            self.assertEqual(
                config["government"]["policy_strategy"], "active_demand"
            )
            self.assertEqual(
                engine._government_strategy(shock_month - 1),
                config["government"]["policy_strategies"]["passive_safety_net"],
            )
            self.assertEqual(
                engine._government_strategy(shock_month),
                config["government"]["policy_strategies"]["active_demand"],
            )

    def test_e2_blocks_ai_attributable_layoffs_and_reduces_work_intensity(
        self,
    ) -> None:
        states = {}
        results = {}
        for scenario in ("E1", "E2"):
            config = scenario_config(load_config(), scenario, 100, 31)
            config["simulation"]["shock_month"] = 1
            config["firms"]["culture_mode"] = "cost_cutter"
            config["firms"]["enable_entry_exit"] = False
            state = initialize_economy(config)
            for firm in state.firms.values():
                firm.ai_multiplier = firm.ai_target
                firm.expected_demand *= 0.80
            results[scenario] = EconomyEngine(state, config)._adjust_employment()
            states[scenario] = state

        self.assertLess(results["E2"][1], results["E1"][1])
        self.assertGreater(
            states["E2"].cumulative_ai_attributable_layoffs_blocked, 0
        )
        self.assertLess(
            mean(
                firm.work_intensity
                for firm in states["E2"].firms.values()
            ),
            1.0,
        )
        employed_hours = [
            resident.monthly_work_hours
            for resident in states["E2"].residents.values()
            if resident.employed
        ]
        self.assertTrue(
            all(
                hours <= float(config["simulation"]["monthly_work_hours"])
                for hours in employed_hours
            )
        )
        self.assertTrue(
            any(
                hours < float(config["simulation"]["monthly_work_hours"])
                for hours in employed_hours
            )
        )

    def test_e2_cost_sharing_and_restructuring_grace_are_auditable(
        self,
    ) -> None:
        config = scenario_config(load_config(), "E2", 100, 33)
        config["simulation"]["shock_month"] = 1
        state = initialize_economy(config)
        for firm in state.firms.values():
            firm.ai_multiplier = firm.ai_target
        engine = EconomyEngine(state, config)
        metric = engine.step()
        self.assertGreater(metric["government_retention_wage_subsidy"], 0.0)
        self.assertAlmostEqual(
            metric["government_retention_wage_subsidy"],
            sum(firm.retention_wage_subsidy for firm in state.firms.values()),
        )
        self.assertAlmostEqual(
            metric["government_retention_wage_subsidy"],
            metric["cumulative_retention_wage_subsidy"],
        )

        candidate = max(
            state.firms.values(), key=lambda firm: len(firm.employee_ids)
        )
        candidate.loss_months = int(
            config["firms"]["bankruptcy_loss_months"]
        )
        candidate.founding_month = 0
        month = int(config["firms"]["bankruptcy_minimum_age_months"]) + 1
        grace = int(
            config["institutional_experiments"]["employment_responsibility"][
                "restructuring_grace_months"
            ]
        )
        for _ in range(grace):
            exits, jobs = engine._process_firm_exits(month)
            self.assertEqual((exits, jobs), (0, 0))
        exits, jobs = engine._process_firm_exits(month)
        self.assertEqual(exits, 1)
        self.assertGreater(jobs, 0)

    def test_e3_levy_closes_earmarked_fund_and_partly_offsets_price_gain(
        self,
    ) -> None:
        paths = {}
        for scenario in ("E1", "E3"):
            config = scenario_config(load_config(), scenario, 100, 37)
            config["simulation"]["shock_month"] = 1
            state = initialize_economy(config)
            engine = EconomyEngine(state, config)
            paths[scenario] = [engine.step() for _ in range(6)]

        e1, e3 = paths["E1"][-1], paths["E3"][-1]
        self.assertEqual(e1["ai_levy_capture_rate"], 0.0)
        self.assertGreater(e3["ai_infrastructure_levy"], 0.0)
        self.assertGreater(e3["ai_levy_rent_base"], e3["ai_infrastructure_levy"])
        self.assertGreater(
            e3["ai_levy_capture_rate"],
            config["institutional_experiments"]["ai_infrastructure_levy"][
                "initial_capture_rate"
            ],
        )
        self.assertLess(
            e3["ai_levy_capture_rate"],
            config["institutional_experiments"]["ai_infrastructure_levy"][
                "capture_rate"
            ],
        )
        self.assertGreater(e3["aggregate_price"], e1["aggregate_price"])
        self.assertAlmostEqual(e3["tax_identity_error"], 0.0, delta=1e-8)
        self.assertAlmostEqual(
            e3["cumulative_ai_levy_revenue"],
            e3["cumulative_ai_levy_public_service_spending"]
            + e3["cumulative_ai_levy_public_investment"]
            + e3["government_ai_levy_fund_balance"],
            delta=1e-6,
        )

    def test_e4_forms_solo_enterprises_without_relabeling_them_unemployed(
        self,
    ) -> None:
        config = scenario_config(load_config(), "E4", 100, 41)
        config["simulation"]["shock_month"] = 1
        solo = config["institutional_experiments"]["solo_enterprise"]
        solo["incubation_readiness_threshold"] = 0.0
        solo["minimum_personal_ai_use_rate"] = 0.0
        solo["startup_cost_months"] = 0.0
        config["banking"]["activation_month"] = 1
        state = initialize_economy(config)
        for resident in state.residents.values():
            resident.cash += 10_000.0
        metric = EconomyEngine(state, config).step()

        self.assertGreater(metric["solo_entries"], 0)
        self.assertGreater(metric["self_employment"], 0)
        self.assertGreater(metric["solo_enterprise_sales"], 0.0)
        self.assertGreater(metric["solo_substitution_sales"], 0.0)
        self.assertGreater(metric["solo_induced_demand_sales"], 0.0)
        self.assertGreater(metric["solo_external_sales"], 0.0)
        self.assertGreaterEqual(metric["solo_b2b_sales"], 0.0)
        self.assertAlmostEqual(
            metric["solo_enterprise_sales"],
            metric["solo_substitution_sales"]
            + metric["solo_b2b_sales"]
            + metric["solo_induced_demand_sales"]
            + metric["solo_external_sales"],
        )
        self.assertEqual(
            metric["wage_employment"]
            + metric["self_employment"]
            + round(metric["unemployment_rate"] * len(state.residents)),
            len(state.residents),
        )
        self.assertAlmostEqual(metric["sales_identity_error"], 0.0, delta=1e-8)

    def test_procurement_scenario_cannot_silently_disable_its_policy_channel(self) -> None:
        config = load_config()
        del config["scenarios"]["E5"]["policy_strategy"]
        del config["scenario_definition_versions"]["institutional_v2"][
            "scenarios"
        ]["E5"]["policy_strategy"]
        with self.assertRaisesRegex(ValueError, "does not assign"):
            scenario_config(config, "E5", 100, 7)

    def test_personal_ai_is_paid_work_input_not_welfare_consumption(self) -> None:
        config = scenario_config(load_config(), "E1", 100, 17)
        config["simulation"]["shock_month"] = 1
        config["firms"]["culture_mode"] = "augmentation"
        state = initialize_economy(config)
        metric = EconomyEngine(state, config).step()
        self.assertGreater(metric["personal_ai_spending"], 0.0)
        self.assertGreater(metric["personal_ai_mean_use_rate"], 0.0)
        resident_spending = sum(
            resident.personal_ai_spending for resident in state.residents.values()
        )
        self.assertAlmostEqual(metric["personal_ai_spending"], resident_spending)
        self.assertAlmostEqual(
            metric["household_consumption"] - metric["real_consumption"] * metric["aggregate_price"],
            resident_spending,
            delta=1e-6,
        )

    def test_procurement_subchannels_never_exceed_total_procurement(self) -> None:
        config = scenario_config(load_config(), "E5", 100, 19)
        config["simulation"]["shock_month"] = 1
        config["government"]["policy_strategy"] = "active_demand_regulation"
        state = initialize_economy(config)
        engine = EconomyEngine(state, config)
        for _ in range(36):
            metric = engine.step()
            support = (
                metric["government_employment_support_procurement"]
                + metric["government_productivity_dividend_procurement"]
            )
            self.assertLessEqual(support, metric["government_procurement"] + 1e-8)

    def test_regulation_penalizes_only_below_cost_market_attraction(self) -> None:
        base = scenario_config(load_config(), "E5", 100, 23)
        base["simulation"]["shock_month"] = 1
        base["firms"]["competition"]["share_adjustment_speed"] = 1.0
        shares = {}
        for strategy in ("active_demand", "active_demand_regulation"):
            config = copy.deepcopy(base)
            config["government"]["policy_strategy"] = strategy
            state = initialize_economy(config)
            state.month = 0
            firms = list(state.firms.values())
            for firm in firms:
                firm.market_share = 1.0 / len(firms)
                firm.ai_multiplier = 1.1
                firm.price = 1.0
                firm.wage_bill = 0.0
                firm.fixed_cost = 0.0
                firm.interest_payment = 0.0
            below_cost = firms[0]
            below_cost.price = 0.8
            below_cost.fixed_cost = 2.0 * below_cost.capacity
            EconomyEngine(state, config)._refresh_capacities_and_shares()
            shares[strategy] = below_cost.market_share
        self.assertLess(
            shares["active_demand_regulation"], shares["active_demand"]
        )

    def test_intents_are_bounded(self) -> None:
        self.assertEqual(bounded_intent("resident", {"consumption_stance": "invalid"}), {"consumption_stance": "normal"})
        self.assertEqual(bounded_intent("firm", {"labor_stance": "patient"}), {"labor_stance": "patient"})
        self.assertEqual(bounded_intent("government", {"policy_stance": "stabilize"}), {"policy_stance": "stabilize"})

    def test_accounting_validation_uses_absolute_and_relative_tolerance(self) -> None:
        metric = {
            "sales_identity_error": 0.0,
            "wage_identity_error": 0.0,
            "tax_identity_error": 0.0,
            "firm_sales": 1.0,
            "gross_wage_bill": 1.0,
            "government_tax_revenue": 1.0,
            "employment_rate": 0.95,
            "aggregate_price": 1.0,
        }
        validate_metric(metric)

        metric["firm_sales"] = 1_000_000_000.0
        metric["sales_identity_error"] = 0.0005
        validate_metric(metric)

        metric["sales_identity_error"] = 0.002
        with self.assertRaises(ValueError):
            validate_metric(metric)

        metric["firm_sales"] = 1.0
        metric["sales_identity_error"] = 2e-8
        with self.assertRaises(ValueError):
            validate_metric(metric)


class ResultLayoutTests(unittest.TestCase):
    def test_cognitive_regimes_are_inferred_from_exact_role_sets(self) -> None:
        self.assertEqual(resolve_cognitive_regime(""), "R0")
        self.assertEqual(resolve_cognitive_regime("government"), "R1")
        self.assertEqual(resolve_cognitive_regime("government,firm"), "R2")
        self.assertEqual(
            resolve_cognitive_regime("resident,firm,government"), "R3"
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_cognitive_regime("government", "R2")

    def test_matrix_cell_path_is_deterministic_and_regular(self) -> None:
        path = matrix_cell_dir(
            root=Path("results/research_matrix"),
            stage="formal",
            scenario_definition_version="institutional_v2",
            cognitive_regime="R2",
            provider="hkust",
            model="gpt-3.5-turbo",
            population=500,
            months=120,
            seed=1,
        )
        self.assertEqual(
            path.as_posix(),
            (
                "results/research_matrix/formal/institutional_v2/"
                "R2_firm_government/hkust_gpt-3-5-turbo/"
                "N00500_M120_S001"
            ),
        )
        aggregate = matrix_aggregate_dir(
            root=Path("results/research_matrix"),
            stage="formal",
            scenario_definition_version="institutional_v2",
            cognitive_regime="R0",
            provider="offline",
            model=None,
            analysis="full_study",
            populations="500,1000",
            seeds="1-50",
        )
        self.assertEqual(
            aggregate.as_posix(),
            (
                "results/research_matrix/formal/institutional_v2/"
                "R0_rules/offline_rules/aggregate/"
                "full_study__P500-1000__S1-50"
            ),
        )


if __name__ == "__main__":
    unittest.main()
