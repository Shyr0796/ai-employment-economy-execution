from __future__ import annotations

import json
from typing import Any

from agentsociety2.env.router_base import RouterBase


class ExecutionRouter(RouterBase):
    """JSON router that never uses an LLM to map bounded economic operations."""

    async def ask(
        self,
        ctx: dict,
        instruction: str,
        readonly: bool = False,
        template_mode: bool = False,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> tuple[dict, str]:
        token = self._set_trace_context(trace_id, parent_span_id, ctx.get("agent_id"))
        try:
            request = json.loads(instruction)
            agent_id = int(ctx.get("agent_id", ctx.get("id")))
            env = self.env_modules[0]
            operation = request.get("op")
            if operation == "observe":
                result = env.observe_agent(agent_id)
            elif operation == "submit_intent":
                if readonly:
                    raise PermissionError("submit_intent requires readonly=False")
                result = env.submit_agent_intent(agent_id, request.get("intent", {}))
            elif operation == "macro":
                result = env.macro_metrics()
            else:
                raise ValueError(f"Unsupported execution operation {operation!r}")
            return ctx, json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        except Exception as exc:
            return ctx, json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False)
        finally:
            self._reset_trace_context(token)

    async def get_world_description(self) -> str:
        return "AI employment economy with resident, firm, and government agents using bounded intents."
