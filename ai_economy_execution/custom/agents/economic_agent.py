from __future__ import annotations

import json
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from agentsociety2.agent.base import AgentBase

from ai_economy_execution.core import bounded_intent
from ai_economy_execution.litellm_compat import install_litellm_compat


# Every Ray worker builds its own LiteLLM Router. Install compatibility hooks
# when this custom module loads, before any worker-local request can start.
install_litellm_compat()


def _classify_llm_failure(error: Exception | str) -> str:
    """Return a stable, provider-neutral failure category for audit gates."""
    text = str(error).lower()
    name = type(error).__name__.lower() if isinstance(error, Exception) else ""
    if (
        "ratelimit" in name
        or "rate limit" in text
        or "exceeded rate limit" in text
        or "too many requests" in text
        or "status code: 429" in text
        or "status_code=429" in text
    ):
        return "rate_limit"
    if "timeout" in name or "timed out" in text or "timeout" in text:
        return "timeout"
    if any(
        marker in f"{name} {text}"
        for marker in (
            "connectionerror",
            "connecterror",
            "networkerror",
            "connection reset",
            "connection refused",
            "name resolution",
            "dns",
            "temporary failure",
        )
    ):
        return "network"
    if "authentication" in name or "unauthorized" in text or "status code: 401" in text:
        return "authentication"
    if "permission" in name or "forbidden" in text or "status code: 403" in text:
        return "permission"
    return "provider_or_unknown"


def _response_provenance(response: Any, requested_model: str | None) -> dict[str, Any]:
    """Capture only safe model-routing identifiers returned by LiteLLM."""
    provenance: dict[str, Any] = {"requested_model": requested_model}
    for source_name, target_name in (
        ("model", "response_model"),
        ("id", "response_id"),
        ("system_fingerprint", "system_fingerprint"),
    ):
        value = getattr(response, source_name, None)
        if value is not None:
            provenance[target_name] = str(value)
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict):
        for key in (
            "litellm_provider",
            "model_id",
            "deployment",
            "deployment_id",
            "azure_deployment",
            "region_name",
            "provider_created_raw",
        ):
            value = hidden.get(key)
            if isinstance(value, (str, int, float, bool)):
                provenance[key] = value
        headers = hidden.get("additional_headers")
        if isinstance(headers, dict):
            safe_headers = {}
            for key in (
                "x-ms-region",
                "x-request-id",
                "apim-request-id",
                "x-ratelimit-limit-requests",
                "x-ratelimit-remaining-requests",
                "x-ratelimit-reset-requests",
            ):
                value = headers.get(key)
                if value is not None:
                    safe_headers[key] = str(value)
            if safe_headers:
                provenance["response_headers"] = safe_headers
    raw_created = getattr(response, "provider_created_raw", None)
    if (
        "provider_created_raw" not in provenance
        and isinstance(raw_created, (str, int, float, bool))
    ):
        provenance["provider_created_raw"] = raw_created
    return provenance


class EconomicAgent(AgentBase):
    """One AgentSociety class dispatching resident, firm, and government roles."""

    @classmethod
    def mcp_description(cls) -> str:
        return "EconomicAgent: resident, firm, or government decision unit with bounded actions."

    async def restore(self, workspace_path: Path, service_proxy: Any) -> None:
        install_litellm_compat()
        await super().restore(workspace_path, service_proxy)
        profile = self.get_profile()
        self._economic_role = str(profile["role"])
        self._economic_id = int(profile["economic_id"])
        self._llm_enabled = bool(self._config.get("llm_enabled", False))

    async def ask(self, message: str, readonly: bool = True, *, t: datetime | None = None) -> str:
        if not self._llm_enabled:
            _, answer = await self.ask_env({}, json.dumps({"op": "observe"}), readonly=True)
            return answer
        response = await self.acompletion(
            [
                {"role": "system", "content": f"You are a {self._economic_role} agent. Answer briefly and never set monetary values."},
                {"role": "user", "content": message},
            ],
            temperature=0,
            # AgentSociety enables LiteLLM's response cache globally. Scientific
            # decision paths must retain one real response per audited request,
            # and HKUST's string-valued `created` field also warns when that
            # cache serializes the response model.
            caching=False,
        )
        return response.choices[0].message.content or ""

    async def step(self, tick: int, t: datetime) -> str:
        _, raw = await self.ask_env({}, json.dumps({"op": "observe"}), readonly=True)
        observation = json.loads(raw)
        if "error" in observation:
            return raw
        if not bool(observation.get("active", True)):
            self._append_decision_audit(
                observation=observation,
                fallback={},
                raw_response=None,
                parsed_action=None,
                final_action={},
                status="inactive",
                fallback_reason="economic_entity_not_active",
                latency_ms=0.0,
                request_hash=None,
                usage=None,
            )
            self._step_count += 1
            self._current_time = t
            return json.dumps(
                {
                    "status": "inactive",
                    "agent_id": self._economic_id,
                },
                ensure_ascii=False,
            )
        intent = self._rule_intent(observation)
        if self._llm_enabled:
            intent = await self._llm_intent(observation, intent)
        else:
            self._append_decision_audit(
                observation=observation,
                fallback=intent,
                raw_response=None,
                parsed_action=intent,
                final_action=bounded_intent(self._economic_role, intent),
                status="rule_only",
                fallback_reason=None,
                latency_ms=0.0,
                request_hash=None,
                usage=None,
            )
        intent = bounded_intent(self._economic_role, intent)
        _, answer = await self.ask_env(
            {}, json.dumps({"op": "submit_intent", "intent": intent}, ensure_ascii=False), readonly=False
        )
        self._step_count += 1
        self._current_time = t
        return answer

    def _rule_intent(self, observation: dict[str, Any]) -> dict[str, str]:
        if self._economic_role == "resident":
            duration = int(
                observation.get("shock_unemployment_duration", 0)
            )
            cash_gap = float(observation.get("cash_gap_months", 0.0))
            income_gap = float(observation.get("income_gap_ratio", 0.0))
            if duration >= 6 or cash_gap < -1.0 or income_gap < -0.20:
                stance = "defensive"
            elif duration >= 3 or cash_gap < -0.5 or income_gap < -0.08:
                stance = "cautious"
            else:
                stance = "normal"
            return {"consumption_stance": stance}
        if self._economic_role == "firm":
            utilization_gap = float(observation.get("utilization_gap", 0.0))
            if float(observation["cash_ratio"]) < 0.25:
                stance = "aggressive"
            elif utilization_gap < -0.13:
                stance = "aggressive"
            elif utilization_gap > 0.07:
                stance = "patient"
            else:
                stance = "baseline"
            return {"labor_stance": stance}
        unemployment_gap = float(observation.get("unemployment_gap", 0.0))
        debt = float(observation["debt_ratio"])
        if debt > 0.50:
            stance = "fiscal_guard"
        elif unemployment_gap > 0.038:
            stance = "stabilize"
        elif unemployment_gap > 0.008:
            stance = "balanced_support"
        else:
            stance = "baseline"
        return {"policy_stance": stance}

    async def _llm_intent(self, observation: dict[str, Any], fallback: dict[str, str]) -> dict[str, Any]:
        allowed = {
            "resident": '{"consumption_stance":"normal|cautious|defensive"}',
            "firm": '{"labor_stance":"patient|baseline|aggressive"}',
            "government": '{"policy_stance":"baseline|stabilize|balanced_support|fiscal_guard"}',
        }[self._economic_role]
        prompt = (
            "The system was calibrated to a pre-shock social equilibrium. Values whose relative "
            "gap is close to zero are normal, not signs of distress. Keep the neutral/baseline "
            "stance unless a relative gap shows material deterioration; do not react merely to "
            "the absolute unemployment rate, utilization level, or cash level. Fields ending in "
            "'_change_3m' are deterministic trailing trends, not forecasts. Use them only when "
            "'trend_available' is true, and change stance only for sustained, material "
            "deterioration. For residents, 'unemployment_duration' is descriptive and includes "
            "the calibrated baseline unemployment stock; do not react to it alone. "
            "'shock_unemployment_duration' is actionable only when 'shock_unemployed' is true.\n"
            f"Observation: {json.dumps(observation, ensure_ascii=False)}\n"
            f"Return JSON only with schema {allowed}. You may choose only one listed categorical value."
        )
        request_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        started = time.perf_counter()
        raw_response: str | None = None
        parsed_action: dict[str, Any] | None = None
        usage: dict[str, Any] | None = None
        fallback_reason: str | None = None
        fallback_category: str | None = None
        response_provenance: dict[str, Any] | None = None
        try:
            response = await self.acompletion(
                [{"role": "system", "content": "Choose a bounded economic stance. Do not invent numbers."}, {"role": "user", "content": prompt}],
                temperature=0,
                caching=False,
            )
            response_provenance = _response_provenance(
                response, getattr(self, "_model_name", None)
            )
            raw_response = response.choices[0].message.content or ""
            response_usage = getattr(response, "usage", None)
            if response_usage is not None:
                if hasattr(response_usage, "model_dump"):
                    usage = response_usage.model_dump()
                elif isinstance(response_usage, dict):
                    usage = response_usage
                else:
                    usage = {
                        key: getattr(response_usage, key)
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                        if hasattr(response_usage, key)
                    }
            start, end = raw_response.find("{"), raw_response.rfind("}")
            if start < 0 or end < start:
                fallback_reason = "missing_json_object"
                fallback_category = "response_parse"
                final_action = bounded_intent(self._economic_role, fallback)
                status = "fallback"
            else:
                parsed = json.loads(raw_response[start : end + 1])
                if not isinstance(parsed, dict):
                    raise ValueError("LLM action must be a JSON object")
                parsed_action = parsed
                final_action = bounded_intent(self._economic_role, parsed_action)
                if final_action != parsed_action:
                    fallback_reason = "invalid_or_out_of_schema_action"
                    fallback_category = "response_schema"
                    status = "bounded"
                else:
                    status = "accepted"
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
            fallback_category = _classify_llm_failure(exc)
            final_action = bounded_intent(self._economic_role, fallback)
            status = "fallback"
        self._append_decision_audit(
            observation=observation,
            fallback=fallback,
            raw_response=raw_response,
            parsed_action=parsed_action,
            final_action=final_action,
            status=status,
            fallback_reason=fallback_reason,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            request_hash=request_hash,
            usage=usage,
            fallback_category=fallback_category,
            response_provenance=response_provenance,
        )
        return final_action

    def _append_decision_audit(
        self,
        *,
        observation: dict[str, Any],
        fallback: dict[str, str],
        raw_response: str | None,
        parsed_action: dict[str, Any] | None,
        final_action: dict[str, str],
        status: str,
        fallback_reason: str | None,
        latency_ms: float,
        request_hash: str | None,
        usage: dict[str, Any] | None,
        fallback_category: str | None = None,
        response_provenance: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "agent_id": self._economic_id,
            "role": self._economic_role,
            "month": int(observation.get("month", -1)) + 1,
            "shock_active": bool(observation.get("shock_active", False)),
            "request_hash": request_hash,
            "observation": observation,
            "fallback_action": fallback,
            "raw_response": raw_response,
            "parsed_action": parsed_action,
            "final_action": final_action,
            "status": status,
            "fallback_reason": fallback_reason,
            "fallback_category": fallback_category,
            "latency_ms": latency_ms,
            "usage": usage,
            "response_provenance": response_provenance,
            "llm_enabled": self._llm_enabled,
        }
        audit_path = self.workspace_root_path() / "decision_audit.jsonl"
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    async def to_workspace(self, workspace_path: Path) -> None:
        self.persist_agent_json(tick=self._step_count, t=self._current_time)
