from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agentsociety2.env.base import EnvBase, tool

from ai_economy_execution.core import EconomyEngine
from ai_economy_execution.metrics import validate_metric
from ai_economy_execution.models import EconomyState


class ExecutionEconomyEnv(EnvBase):
    """Shared monthly economy clearing environment for every economic agent."""

    def __init__(self, state: dict[str, Any] | EconomyState | None = None, config: dict[str, Any] | None = None):
        super().__init__()
        if config is None:
            from ai_economy_execution.configuration import load_config, scenario_config

            config = scenario_config(load_config(), "E0")
        if state is None:
            from ai_economy_execution.initialization import initialize_economy

            state = initialize_economy(config)
        self.state = state if isinstance(state, EconomyState) else EconomyState.from_dict(state)
        self.config = config
        self.engine = EconomyEngine(self.state, config)

    @classmethod
    def init_description(cls) -> str:
        return "ExecutionEconomyEnv(state=<serialized EconomyState>, config=<baseline configuration>)"

    @tool(readonly=True, kind="observe")
    def observe_agent(self, agent_id: int) -> dict[str, Any]:
        try:
            observation = self.engine.observe(agent_id)
        except KeyError:
            # AgentSociety keeps a fixed roster.  Firm slots that have not
            # entered yet, or firms that have exited, remain dormant until the
            # economic core contains their id again.
            if 10000 < int(agent_id) < int(self.state.government.id):
                return {
                    "agent_id": int(agent_id),
                    "role": "firm",
                    "active": False,
                    "month": int(self.state.month),
                    "shock_active": bool(
                        self.state.month
                        >= int(self.config["simulation"]["shock_month"])
                    ),
                }
            raise
        observation["active"] = True
        return observation

    @tool(readonly=False)
    def submit_agent_intent(self, agent_id: int, intent: dict[str, Any]) -> dict[str, str]:
        return self.engine.submit_intent(agent_id, intent)

    @tool(readonly=True, kind="statistics")
    def macro_metrics(self) -> dict[str, Any]:
        return self.engine.current_macro()

    async def step(self, tick: int, t: datetime):
        self.t = t
        expected_ids = set(self.state.residents) | set(self.state.firms) | {self.state.government.id}
        missing = expected_ids - set(self.state.intents)
        if missing:
            raise RuntimeError(
                f"Economic settlement blocked: {len(missing)} agents did not submit an intent; "
                f"sample={sorted(missing)[:5]}"
            )
        submitted = len(self.state.intents)
        metric = self.engine.step()
        metric["submitted_agent_intents"] = submitted
        validate_metric(metric)
        return metric

    async def to_workspace(self, workspace_path: Path | None = None) -> None:
        await super().to_workspace(workspace_path)
        if self._workspace_root is None:
            return
        target = self._workspace_root / "economy_state.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state.to_dict(), ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)

    async def restore(self, workspace_path: Path) -> bool:
        await super().restore(workspace_path)
        target = Path(workspace_path) / "economy_state.json"
        if not target.exists():
            return False
        self.state = EconomyState.from_dict(json.loads(target.read_text(encoding="utf-8")))
        self.engine = EconomyEngine(self.state, self.config)
        return True
