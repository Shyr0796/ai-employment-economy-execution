from __future__ import annotations

import os
import json
import tempfile
import unittest
import asyncio
import logging
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class AgentSocietyModuleTests(unittest.TestCase):
    def test_terminal_metadata_and_manifest_are_authoritative(self) -> None:
        from ai_economy_execution.run import (
            _finalize_step_metadata,
            _write_run_manifest,
        )

        with tempfile.TemporaryDirectory(prefix="run-manifest-test-") as temp_dir:
            output = Path(temp_dir)
            (output / "SOCIETY_STEP.json").write_text(
                json.dumps({
                    "step_count": 120,
                    "completed_step_count": 0,
                    "terminated": False,
                }),
                encoding="utf-8",
            )
            resolved = output / "resolved_config.json"
            resolved.write_text(json.dumps({"config": {"simulation": {"months": 120}}}), encoding="utf-8")
            step = _finalize_step_metadata(
                output,
                status="completed",
                requested_steps=120,
                completed_steps=120,
                final_model_month=120,
            )
            manifest = _write_run_manifest(
                output,
                status="completed",
                started_at="2026-07-21T00:00:00+00:00",
                source_fingerprint="abc123",
                resolved_config_path=resolved,
                scenario="E5",
                population=500,
                seed=1,
                starting_month=0,
                requested_final_month=120,
                completed_history_months=120,
                completed_steps=120,
                decision_audit={"records": 63720, "closed": True},
            )
            persisted_step = json.loads((output / "SOCIETY_STEP.json").read_text(encoding="utf-8"))
            persisted_manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))

        self.assertTrue(step["terminated"])
        self.assertEqual(step["completed_step_count"], 120)
        self.assertEqual(persisted_step["status"], "completed")
        self.assertTrue(manifest["completion"]["complete"])
        self.assertEqual(persisted_manifest["decision_audit"]["records"], 63720)

    def test_decision_audit_aggregation_counts_statuses(self) -> None:
        from ai_economy_execution.run import _aggregate_decision_audits

        with tempfile.TemporaryDirectory(prefix="decision-audit-test-") as temp_dir:
            output = Path(temp_dir)
            for agent, records in (
                (
                    "agent_1",
                    (
                        {
                            "status": "accepted",
                            "llm_enabled": True,
                            "role": "resident",
                            "response_provenance": {
                                "requested_model": "gpt-3.5-turbo",
                                "response_model": "gpt-3.5-turbo",
                            },
                        },
                        {
                            "status": "fallback",
                            "llm_enabled": True,
                            "role": "resident",
                            "fallback_category": "rate_limit",
                        },
                    ),
                ),
                (
                    "agent_2",
                    (
                        {
                            "status": "bounded",
                            "llm_enabled": True,
                            "role": "firm",
                            "response_provenance": {
                                "requested_model": "gpt-3.5-turbo",
                                "response_model": "gpt-4o-mini",
                            },
                        },
                        {"status": "rule_only", "llm_enabled": False},
                        {"status": "inactive", "llm_enabled": True},
                    ),
                ),
            ):
                workspace = output / "agents" / agent
                workspace.mkdir(parents=True)
                (workspace / "decision_audit.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in records),
                    encoding="utf-8",
                )
            counts = _aggregate_decision_audits(output)
            self.assertEqual(counts["records"], 5)
            self.assertEqual(counts["accepted"], 1)
            self.assertEqual(counts["fallbacks"], 1)
            self.assertEqual(counts["bounded"], 1)
            self.assertEqual(counts["rule_only"], 1)
            self.assertEqual(counts["inactive"], 1)
            self.assertEqual(counts["llm_eligible_records"], 3)
            self.assertEqual(counts["unknown_status"], 0)
            self.assertAlmostEqual(counts["fallback_rate"], 1 / 3)
            self.assertEqual(counts["fallback_categories"], {"rate_limit": 1})
            self.assertEqual(
                counts["model_provenance"]["mismatched_response_model"], 1
            )
            self.assertEqual(
                counts["model_provenance"]["pairs"],
                {
                    "gpt-3.5-turbo -> gpt-3.5-turbo": 1,
                    "gpt-3.5-turbo -> gpt-4o-mini": 1,
                },
            )

    def test_decision_quality_gate_is_role_aware_and_checks_model_identity(self) -> None:
        from ai_economy_execution.run import _decision_quality_gate

        audit = {
            "fallback_rate": 0.02,
            "by_role": {
                "resident": {
                    "llm_eligible_records": 100,
                    "fallback_rate": 0.01,
                },
                "firm": {
                    "llm_eligible_records": 20,
                    "fallback_rate": 0.10,
                },
            },
            "model_provenance": {
                "missing_response_model": 0,
                "mismatched_response_model": 2,
            },
        }
        gate = _decision_quality_gate(
            audit,
            max_fallback_rate=0.05,
            max_role_fallback_rate=0.05,
            require_response_model_match=True,
        )

        self.assertFalse(gate["pass"])
        self.assertTrue(any("firm fallback rate" in item for item in gate["violations"]))
        self.assertTrue(any("different response model" in item for item in gate["violations"]))

    def test_hkust_documented_model_alias_passes_identity_audit(self) -> None:
        from ai_economy_execution.run import (
            _aggregate_decision_audits,
            _decision_quality_gate,
        )

        with tempfile.TemporaryDirectory(prefix="hkust-model-alias-test-") as temp_dir:
            output = Path(temp_dir)
            workspace = output / "agents" / "agent_1"
            workspace.mkdir(parents=True)
            (workspace / "decision_audit.jsonl").write_text(
                json.dumps(
                    {
                        "status": "accepted",
                        "llm_enabled": True,
                        "role": "resident",
                        "response_provenance": {
                            "requested_model": "gpt-3.5-turbo",
                            "response_model": "gpt-4o-mini-2024-07-18",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            audit = _aggregate_decision_audits(output, provider="hkust")

        self.assertEqual(
            audit["model_provenance"]["aliased_response_model"],
            1,
        )
        self.assertEqual(
            audit["model_provenance"]["mismatched_response_model"],
            0,
        )
        gate = _decision_quality_gate(
            audit,
            max_fallback_rate=0.01,
            max_role_fallback_rate=0.01,
            require_response_model_match=True,
        )
        self.assertTrue(gate["pass"])

    def test_hkust_alias_does_not_relax_other_providers(self) -> None:
        from ai_economy_execution.providers import response_model_match_kind

        self.assertEqual(
            response_model_match_kind(
                "openai",
                "gpt-3.5-turbo",
                "gpt-4o-mini-2024-07-18",
            ),
            "mismatch",
        )
        self.assertEqual(
            response_model_match_kind(
                "hkust",
                "gpt-4",
                "gpt-4o-mini-2024-07-18",
            ),
            "mismatch",
        )

    def test_llm_failure_classification_and_response_provenance(self) -> None:
        from ai_economy_execution.custom.agents.economic_agent import (
            _classify_llm_failure,
            _response_provenance,
        )

        self.assertEqual(
            _classify_llm_failure(
                "RateLimitError: requests exceeded rate limit; status code: 429"
            ),
            "rate_limit",
        )
        self.assertEqual(
            _classify_llm_failure("ConnectError: connection reset by peer"),
            "network",
        )
        response = SimpleNamespace(
            model="gpt-4o-mini",
            id="request-123",
            system_fingerprint="fp-test",
            _hidden_params={
                "litellm_provider": "openai",
                "deployment": "eastus-deployment",
                "additional_headers": {
                    "x-ms-region": "eastus",
                    "authorization": "must-not-be-recorded",
                },
            },
        )
        provenance = _response_provenance(response, "gpt-3.5-turbo")

        self.assertEqual(provenance["requested_model"], "gpt-3.5-turbo")
        self.assertEqual(provenance["response_model"], "gpt-4o-mini")
        self.assertEqual(provenance["deployment"], "eastus-deployment")
        self.assertEqual(
            provenance["response_headers"], {"x-ms-region": "eastus"}
        )

    def test_structural_unemployment_duration_does_not_trigger_shock_rule(self) -> None:
        from ai_economy_execution.custom.agents.economic_agent import (
            EconomicAgent,
        )

        agent = object.__new__(EconomicAgent)
        agent._economic_role = "resident"
        observation = {
            "unemployment_duration": 12,
            "shock_unemployment_duration": 0,
            "cash_gap_months": 0.0,
            "income_gap_ratio": 0.0,
        }

        self.assertEqual(
            agent._rule_intent(observation),
            {"consumption_stance": "normal"},
        )
        observation["shock_unemployment_duration"] = 6
        self.assertEqual(
            agent._rule_intent(observation),
            {"consumption_stance": "defensive"},
        )

    def test_hkust_string_created_is_normalized_before_serialization(self) -> None:
        from litellm.llms.aiohttp_openai.chat.transformation import (
            AiohttpOpenAIChatConfig,
        )
        from litellm.types.utils import ModelResponse

        with patch.dict(
            os.environ,
            {
                "AGENTSOCIETY_LLM_API_KEY": "test-key",
                "AGENTSOCIETY_LLM_API_BASE": "http://127.0.0.1:1/v1",
                "AGENTSOCIETY_LLM_MODEL": "test-model",
            },
        ):
            from ai_economy_execution.custom.agents.economic_agent import (
                _response_provenance,
            )
        from ai_economy_execution.litellm_compat import install_litellm_compat

        class RawResponse:
            async def json(self) -> dict:
                return {
                    "id": "request-created-test",
                    "object": "chat.completion",
                    "model": "gpt-4o-mini-2024-07-18",
                    "created": "2026-07-25 11:43:50",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": '{"consumption_stance":"normal"}',
                            },
                        }
                    ],
                }

        async def transform() -> ModelResponse:
            install_litellm_compat()
            return await AiohttpOpenAIChatConfig().transform_response(
                model="gpt-3.5-turbo",
                raw_response=RawResponse(),
                model_response=ModelResponse(),
                logging_obj=None,
                request_data={},
                messages=[],
                optional_params={},
                litellm_params={},
                encoding=None,
            )

        response = asyncio.run(transform())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            response.model_dump()

        self.assertIsInstance(response.created, int)
        self.assertFalse(
            any("field_name='created'" in str(item.message) for item in caught)
        )
        provenance = _response_provenance(response, "gpt-3.5-turbo")
        self.assertEqual(
            provenance["provider_created_raw"],
            "2026-07-25 11:43:50",
        )

    def test_agentsociety_router_normalizes_hkust_created_before_logging(self) -> None:
        from aiohttp import web

        with patch.dict(
            os.environ,
            {
                "AGENTSOCIETY_LLM_API_KEY": "test-key",
                "AGENTSOCIETY_LLM_API_BASE": "http://127.0.0.1:1/v1",
                "AGENTSOCIETY_LLM_MODEL": "test-model",
            },
        ):
            from agentsociety2.config.llm_dispatcher import LLMClient

        from ai_economy_execution.litellm_compat import install_litellm_compat

        async def exercise_router() -> tuple[object, list[warnings.WarningMessage]]:
            async def completion(_: web.Request) -> web.Response:
                return web.json_response(
                    {
                        "id": "router-created-test",
                        "object": "chat.completion",
                        "model": "gpt-4o-mini-2024-07-18",
                        "created": "2026-07-25 11:43:50",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {
                                    "role": "assistant",
                                    "content": '{"consumption_stance":"normal"}',
                                },
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 8,
                            "completion_tokens": 4,
                            "total_tokens": 12,
                        },
                    }
                )

            app = web.Application()
            app.router.add_post("/v1/chat/completions", completion)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            sockets = site._server.sockets
            assert sockets
            port = int(sockets[0].getsockname()[1])
            try:
                install_litellm_compat()
                client = LLMClient(
                    model_name="gpt-3.5-turbo-router-test",
                    base_url=f"http://127.0.0.1:{port}/v1",
                    api_key="test-key",
                    model_type="resident",
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    response = await client.call(
                        messages=[{"role": "user", "content": "test"}],
                        max_retries=1,
                        caching=False,
                    )
                    response.model_dump()
                return response, caught
            finally:
                await runner.cleanup()

        response, caught = asyncio.run(exercise_router())
        self.assertIsInstance(response.created, int)
        self.assertEqual(
            getattr(response, "provider_created_raw", None),
            "2026-07-25 11:43:50",
        )
        self.assertFalse(
            any("field_name='created'" in str(item.message) for item in caught)
        )

    def test_litellm_noise_filter_keeps_actionable_warnings(self) -> None:
        from ai_economy_execution.litellm_compat import install_litellm_compat

        install_litellm_compat()
        logger = logging.getLogger("LiteLLM")
        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        previous_handlers = list(logger.handlers)
        logger.handlers = [Capture()]
        try:
            logger.warning(
                "register_model: model=test not in built-in cost map and no "
                "prefix/region variant matched; cache cost fields will default to 0"
            )
            logger.warning("Rate limit for test model")
        finally:
            logger.handlers = previous_handlers

        messages = [record.getMessage() for record in records]
        self.assertEqual(messages, ["Rate limit for test model"])

    def test_future_firm_slots_cover_dynamic_entry_upper_bound(self) -> None:
        from ai_economy_execution.configuration import load_config, scenario_config
        from ai_economy_execution.initialization import (
            build_agent_specs,
            initialize_economy,
        )
        from ai_economy_execution.run import _reserve_future_firm_agent_specs

        config = scenario_config(load_config(), "E5", 500, 1)
        state = initialize_economy(config)
        specs = build_agent_specs(state, {"firm"})
        original_ids = {int(spec["id"]) for spec in specs}
        specs, roster = _reserve_future_firm_agent_specs(
            specs,
            state,
            {"firm"},
            config,
            total_months=120,
        )
        all_ids = [int(spec["id"]) for spec in specs]
        expected = 120 * int(config["firms"]["max_monthly_entries"])

        self.assertEqual(roster["reserved_count"], expected)
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertTrue(original_ids.issubset(all_ids))
        self.assertEqual(roster["first_reserved_id"], state.next_firm_id)
        self.assertEqual(
            roster["last_reserved_id"],
            state.next_firm_id + expected - 1,
        )
        reserved = [
            spec
            for spec in specs
            if int(spec["id"]) >= int(state.next_firm_id)
            and int(spec["id"]) < int(state.government.id)
        ]
        self.assertTrue(all(spec["config"]["llm_enabled"] for spec in reserved))
        self.assertTrue(all(spec["config"]["lifecycle_slot"] for spec in reserved))

    def test_environment_returns_inactive_observation_for_future_firm_slot(self) -> None:
        from ai_economy_execution.configuration import load_config, scenario_config
        from ai_economy_execution.custom.envs.execution_economy_env import (
            ExecutionEconomyEnv,
        )
        from ai_economy_execution.initialization import initialize_economy

        config = scenario_config(load_config(), "E5", 50, 1)
        state = initialize_economy(config)
        environment = ExecutionEconomyEnv(state.to_dict(), config)
        observation = environment.observe_agent(state.next_firm_id)

        self.assertEqual(observation["role"], "firm")
        self.assertFalse(observation["active"])
        self.assertEqual(observation["agent_id"], state.next_firm_id)

    def test_behavior_audit_writes_conditioned_actions_and_direction_checks(self) -> None:
        from ai_economy_execution.behavior_audit import write_behavior_audit

        records = [
            {
                "role": "firm",
                "status": "inactive",
                "observation": {"active": False},
                "final_action": {},
            },
            {
                "role": "resident",
                "status": "accepted",
                "observation": {
                    "unemployment_duration": 0,
                    "shock_unemployment_duration": 0,
                    "income_gap_ratio": 0.0,
                    "cash_gap_months": 0.0,
                },
                "final_action": {"consumption_stance": "normal"},
            },
            {
                "role": "resident",
                "status": "accepted",
                "observation": {
                    "unemployment_duration": 6,
                    "shock_unemployment_duration": 6,
                    "income_gap_ratio": -0.30,
                    "cash_gap_months": -2.0,
                },
                "final_action": {"consumption_stance": "defensive"},
            },
            {
                "role": "firm",
                "status": "accepted",
                "observation": {"utilization_gap": 0.0, "cash_ratio": 1.0},
                "final_action": {"labor_stance": "baseline"},
            },
            {
                "role": "firm",
                "status": "accepted",
                "observation": {"utilization_gap": -0.20, "cash_ratio": 1.0},
                "final_action": {"labor_stance": "aggressive"},
            },
            {
                "role": "government",
                "status": "accepted",
                "observation": {"unemployment_gap": 0.0, "debt_ratio": 0.10},
                "final_action": {"policy_stance": "baseline"},
            },
            {
                "role": "government",
                "status": "accepted",
                "observation": {"unemployment_gap": 0.05, "debt_ratio": 0.10},
                "final_action": {"policy_stance": "stabilize"},
            },
            {
                "role": "government",
                "status": "accepted",
                "observation": {"unemployment_gap": 0.0, "debt_ratio": 0.60},
                "final_action": {"policy_stance": "fiscal_guard"},
            },
        ]
        with tempfile.TemporaryDirectory(prefix="behavior-audit-test-") as temp_dir:
            output = Path(temp_dir)
            source = output / "decision_audit.jsonl"
            source.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            result = write_behavior_audit(source, output)

            self.assertTrue((output / "decision_behavior_summary.json").is_file())
            self.assertTrue((output / "decision_behavior_summary.csv").is_file())
            self.assertEqual(result["records"], 8)
            self.assertEqual(result["active_records"], 7)
            self.assertEqual(
                result["by_role"],
                {"firm": 3, "government": 3, "resident": 2},
            )
            self.assertNotIn("<missing>", result["action_distributions"]["firm"])
            self.assertEqual(
                result["action_distributions"]["resident"],
                {"defensive": 1, "normal": 1},
            )
            checks = result["monotonicity_checks"]
            self.assertEqual(len(checks), 5)
            self.assertTrue(all(check["available"] for check in checks))
            self.assertTrue(
                all(check["passes_direction_check"] for check in checks)
            )
            qualification = result["behavior_qualification"]
            self.assertTrue(qualification["suitable_for_behavior_claims"])
            self.assertEqual(qualification["warnings"], [])

    def test_behavior_qualification_flags_single_action_and_missing_stress(self) -> None:
        from ai_economy_execution.behavior_audit import (
            summarize_decision_records,
        )

        result = summarize_decision_records(
            [
                {
                    "role": "resident",
                    "status": "accepted",
                    "observation": {
                        "unemployment_duration": 0,
                        "shock_unemployment_duration": 0,
                        "income_gap_ratio": 0.0,
                        "cash_gap_months": 0.0,
                    },
                    "final_action": {"consumption_stance": "normal"},
                }
            ]
        )

        qualification = result["behavior_qualification"]
        self.assertFalse(qualification["suitable_for_behavior_claims"])
        self.assertEqual(qualification["single_action_roles"], ["resident"])
        self.assertIn(
            "monotonicity_coverage_incomplete",
            qualification["warnings"],
        )
        self.assertIn(
            "single_action_role_detected",
            qualification["warnings"],
        )

    def test_behavior_qualification_excludes_rule_only_roles(self) -> None:
        from ai_economy_execution.behavior_audit import (
            summarize_decision_records,
        )

        result = summarize_decision_records(
            [
                {
                    "role": "resident",
                    "status": "rule_only",
                    "observation": {"income_gap_ratio": 0.0},
                    "final_action": {"consumption_stance": "normal"},
                },
                {
                    "role": "firm",
                    "status": "rule_only",
                    "observation": {"utilization_gap": 0.0},
                    "final_action": {"labor_stance": "baseline"},
                },
                {
                    "role": "government",
                    "status": "accepted",
                    "observation": {
                        "unemployment_gap": 0.0,
                        "debt_ratio": 0.10,
                    },
                    "final_action": {"policy_stance": "baseline"},
                },
                {
                    "role": "government",
                    "status": "accepted",
                    "observation": {
                        "unemployment_gap": 0.05,
                        "debt_ratio": 0.60,
                    },
                    "final_action": {"policy_stance": "stabilize"},
                },
                {
                    "role": "government",
                    "status": "accepted",
                    "observation": {
                        "unemployment_gap": 0.0,
                        "debt_ratio": 0.60,
                    },
                    "final_action": {"policy_stance": "fiscal_guard"},
                },
            ]
        )
        qualification = result["behavior_qualification"]
        self.assertEqual(qualification["qualified_roles"], ["government"])
        self.assertEqual(
            qualification["excluded_non_llm_roles"], ["firm", "resident"]
        )
        self.assertEqual(qualification["total_monotonicity_checks"], 2)
        self.assertEqual(qualification["single_action_roles"], [])
        self.assertTrue(qualification["suitable_for_behavior_claims"])

    def test_provider_configuration_is_network_free_and_validates_inputs(self) -> None:
        from ai_economy_execution.providers import configure_agentsociety

        with patch.dict(os.environ, {}, clear=False):
            offline = configure_agentsociety("offline")
            self.assertEqual(offline["provider"], "offline")
            self.assertEqual(offline["model"], "offline-rule-mode")
            self.assertEqual(
                os.environ["AGENTSOCIETY_LLM_API_BASE"],
                "http://127.0.0.1:1/v1",
            )
            self.assertEqual(os.environ["LITELLM_LOCAL_MODEL_COST_MAP"], "True")
        with self.assertRaisesRegex(ValueError, "custom provider requires"):
            configure_agentsociety("custom")
        with self.assertRaisesRegex(ValueError, "Unknown provider"):
            configure_agentsociety("not-a-provider")
        with patch.dict(os.environ, {"TEST_API_KEY": "test-no-network"}):
            custom = configure_agentsociety(
                "custom",
                key_env="TEST_API_KEY",
                api_base="http://127.0.0.1:1/v1",
                model="test-model",
            )
        self.assertEqual(
            custom,
            {
                "provider": "custom",
                "api_base": "http://127.0.0.1:1/v1",
                "model": "test-model",
            },
        )

    def test_api_preflight_registers_modules_without_network(self) -> None:
        from ai_economy_execution.api_preflight import parse_roles, run_preflight

        with self.assertRaisesRegex(ValueError, "Unknown LLM roles"):
            parse_roles("resident,household")
        with patch(
            "socket.socket.connect",
            side_effect=AssertionError("preflight attempted a network connection"),
        ):
            result = run_preflight(
                provider="offline",
                llm_roles=parse_roles("government,resident"),
            )
        self.assertEqual(result["status"], "ready")
        self.assertFalse(result["network_called"])
        self.assertEqual(result["credential_check"], "not_required")
        self.assertEqual(result["llm_roles"], ["government", "resident"])
        self.assertEqual(result["registered_agents"], ["EconomicAgent"])
        self.assertEqual(
            result["registered_environments"], ["ExecutionEconomyEnv"]
        )

    def test_custom_modules_are_valid(self) -> None:
        os.environ.setdefault("AGENTSOCIETY_LLM_API_KEY", "test-no-network")
        os.environ.setdefault("AGENTSOCIETY_LLM_API_BASE", "http://127.0.0.1:1/v1")
        os.environ.setdefault("AGENTSOCIETY_LLM_MODEL", "test-rule-mode")
        package_root = Path(__file__).resolve().parents[1]
        os.environ["WORKSPACE_PATH"] = str(package_root)
        from agentsociety2.registry import scan_and_register_custom_modules

        result = scan_and_register_custom_modules(package_root)
        self.assertFalse(result.get("errors"))
        self.assertFalse(result.get("registration_errors"))
        self.assertEqual([item["class_name"] for item in result["agents"]], ["EconomicAgent"])
        self.assertEqual([item["class_name"] for item in result["envs"]], ["ExecutionEconomyEnv"])

    def test_qualified_checkpoint_restores_as_a_new_scenario_branch(self) -> None:
        from ai_economy_execution.configuration import load_config, scenario_config
        from ai_economy_execution.core import EconomyEngine
        from ai_economy_execution.initialization import initialize_economy
        from ai_economy_execution.run import (
            _pre_equilibrium_config_fingerprint,
            _restore_pre_equilibrium_checkpoint,
            _source_fingerprint,
        )

        origin_config = scenario_config(load_config(), "E0", 10, 3)
        target_config = scenario_config(load_config(), "E5", 10, 3)
        state = initialize_economy(origin_config)
        engine = EconomyEngine(state, origin_config)
        for _ in range(int(origin_config["simulation"]["warmup_months"])):
            engine.step()
        provider = {
            "provider": "offline",
            "api_base": "http://127.0.0.1:1/v1",
            "model": "offline-rule-mode",
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = directory / "pre_equilibrium_state.json"
            checkpoint.write_text(json.dumps(state.to_dict()), encoding="utf-8")
            (directory / "pre_equilibrium_audit.json").write_text(
                json.dumps({"path_gate_pass": True}), encoding="utf-8"
            )
            (directory / "resolved_config.json").write_text(
                json.dumps({
                    "execution": {
                        "scenario": "E0",
                        "llm_roles": [],
                        "provider": provider,
                        "source_fingerprint": _source_fingerprint(),
                        "pre_equilibrium_config_fingerprint": (
                            _pre_equilibrium_config_fingerprint(origin_config)
                        ),
                    }
                }),
                encoding="utf-8",
            )
            restored, audit, lineage = _restore_pre_equilibrium_checkpoint(
                checkpoint,
                config=target_config,
                scenario="E5",
                llm_roles=set(),
                provider_info=provider,
                source_fingerprint=_source_fingerprint(),
                allow_unstable=False,
                allow_source_mismatch=False,
            )
            hkust_provider = {
                "provider": "hkust",
                "api_base": "https://gpt-api.hkust-gz.edu.cn/v1",
                "model": "gpt-3.5-turbo",
            }
            with self.assertRaisesRegex(ValueError, "LLM roles differ"):
                _restore_pre_equilibrium_checkpoint(
                    checkpoint,
                    config=target_config,
                    scenario="E5",
                    llm_roles={"government"},
                    provider_info=hkust_provider,
                    source_fingerprint=_source_fingerprint(),
                    allow_unstable=False,
                    allow_source_mismatch=False,
                )
            activated, _, activated_lineage = _restore_pre_equilibrium_checkpoint(
                checkpoint,
                config=target_config,
                scenario="E5",
                llm_roles={"government"},
                provider_info=hkust_provider,
                source_fingerprint=_source_fingerprint(),
                allow_unstable=False,
                allow_source_mismatch=False,
                allow_cognitive_activation=True,
            )
        self.assertTrue(audit["path_gate_pass"])
        self.assertEqual(restored.month, 24)
        self.assertEqual(restored.scenario, "E5")
        self.assertTrue(all(row["scenario"] == "E5" for row in restored.history))
        self.assertEqual(lineage["origin_scenario"], "E0")
        self.assertEqual(activated.month, 24)
        self.assertEqual(
            activated_lineage["cognitive_activation"]["activation_month"], 25
        )
        self.assertEqual(
            activated_lineage["cognitive_activation"]["to_llm_roles"],
            ["government"],
        )


if __name__ == "__main__":
    unittest.main()
