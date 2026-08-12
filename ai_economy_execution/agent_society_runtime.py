from __future__ import annotations

from pathlib import Path
from typing import Any


_ACTOR_CLASS = None


def _get_actor_class():
    global _ACTOR_CLASS
    if _ACTOR_CLASS is not None:
        return _ACTOR_CLASS
    import ray

    @ray.remote(max_concurrency=128)
    class ExecutionEnvironmentActor:
        def __init__(self, state: dict[str, Any], config: dict[str, Any], run_dir: str):
            from ai_economy_execution.custom.envs.execution_economy_env import ExecutionEconomyEnv
            from ai_economy_execution.router import ExecutionRouter

            self.env = ExecutionEconomyEnv(state=state, config=config)
            self.router = ExecutionRouter(env_modules=[self.env])
            self.router.run_dir = Path(run_dir)
            self.router.bind_env_workspaces(Path(run_dir) / "env", ["ExecutionEconomyEnv"])

        async def ask(self, ctx, instruction, readonly=False, template_mode=False, trace_id=None, parent_span_id=None):
            return await self.router.ask(ctx, instruction, readonly, template_mode, trace_id, parent_span_id)

        async def init(self, start_datetime):
            return await self.router.init(start_datetime)

        async def step(self, tick, t):
            return await self.router.step(tick, t)

        async def to_workspaces(self):
            return await self.router.to_workspaces()

        async def from_workspaces(self):
            return await self.router.from_workspaces()

        async def close(self):
            return await self.router.close()

        def set_current_time(self, t):
            self.router.set_current_time(t)

        def set_replay_writer(self, writer):
            self.router.set_replay_writer(writer)

        async def get_world_description(self):
            return await self.router.get_world_description()

        def get_history(self):
            return self.env.state.history

        def get_state(self):
            return self.env.state.to_dict()

    _ACTOR_CLASS = ExecutionEnvironmentActor
    return _ACTOR_CLASS


def create_environment_proxy(state: dict[str, Any], config: dict[str, Any], run_dir: str | Path):
    import ray
    from agentsociety2.env.env_router_proxy import EnvRouterProxy

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    actor = _get_actor_class().remote(state, config, str(Path(run_dir).resolve()))
    proxy = EnvRouterProxy(actor, run_dir=str(Path(run_dir).resolve()), env_module_types=[])
    return proxy, actor
