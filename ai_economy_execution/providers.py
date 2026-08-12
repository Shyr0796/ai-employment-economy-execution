from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Provider:
    key_env: str
    api_base: str
    model: str


PROVIDERS = {
    "hkust": Provider("HKUST_API_KEY", "https://gpt-api.hkust-gz.edu.cn/v1", "gpt-4"),
    "deepseek": Provider("DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-chat"),
    "dashscope": Provider("DASHSCOPE_API_KEY", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
    "moonshot": Provider("MOONSHOT_API_KEY", "https://api.moonshot.cn/v1", "moonshot-v1-8k"),
    "openai": Provider("OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-4.1-mini"),
}

# The HKUST gateway exposes legacy request aliases while reporting the concrete
# model that served the response. Keep this mapping explicit: it lets the audit
# gate accept documented routing without weakening checks for other providers.
RESPONSE_MODEL_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "hkust": {
        "gpt-3.5-turbo": ("gpt-4o-mini",),
        "gpt-4": ("gpt-4o",),
    },
}


def _normalized_model_name(model: str) -> str:
    return str(model).removeprefix("openai/")


def _matches_versioned_model(response_model: str, expected_model: str) -> bool:
    """Match an exact model or its YYYY-MM-DD pinned release."""
    return bool(
        re.fullmatch(
            rf"{re.escape(expected_model)}(?:-\d{{4}}-\d{{2}}-\d{{2}})?",
            response_model,
        )
    )


def response_model_match_kind(
    provider: str | None,
    requested_model: str,
    response_model: str,
) -> str:
    """Classify response identity as exact, a documented alias, or mismatch."""
    requested = _normalized_model_name(requested_model)
    response = _normalized_model_name(response_model)
    if requested == response:
        return "exact"
    aliases = RESPONSE_MODEL_ALIASES.get(str(provider), {}).get(requested, ())
    if any(_matches_versioned_model(response, alias) for alias in aliases):
        return "alias"
    return "mismatch"


def response_model_matching_policy(provider: str | None) -> dict[str, object]:
    """Return the key-free matching policy persisted with decision audits."""
    aliases = RESPONSE_MODEL_ALIASES.get(str(provider), {})
    return {
        "provider": provider,
        "mode": "exact_or_documented_provider_alias" if aliases else "exact",
        "aliases": {key: list(values) for key, values in sorted(aliases.items())},
    }


def load_dotenv_if_available(path: str | Path | None = None) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    candidate = Path(path) if path else Path(__file__).resolve().parents[1] / ".env"
    if candidate.exists():
        load_dotenv(candidate, override=False)


def configure_agentsociety(
    provider: str = "offline",
    *,
    key_env: str | None = None,
    api_base: str | None = None,
    model: str | None = None,
) -> dict[str, str]:
    """Map the prior OpenAI-compatible provider convention to AgentSociety."""
    os.environ.setdefault("MEM0_TELEMETRY", "False")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    if provider == "offline":
        resolved = {"api_key": "offline-no-network", "api_base": "http://127.0.0.1:1/v1", "model": "offline-rule-mode"}
    else:
        load_dotenv_if_available()
        if provider == "custom":
            if not key_env or not api_base or not model:
                raise ValueError("custom provider requires key_env, api_base, and model")
            spec = Provider(key_env, api_base, model)
        else:
            try:
                default_spec = PROVIDERS[provider]
            except KeyError as exc:
                raise ValueError(f"Unknown provider {provider!r}") from exc
            spec = Provider(
                key_env or default_spec.key_env,
                api_base or default_spec.api_base,
                model or default_spec.model,
            )
        key = os.getenv(spec.key_env)
        if not key:
            raise ValueError(f"Environment variable {spec.key_env} is required for provider {provider}")
        resolved = {"api_key": key, "api_base": spec.api_base, "model": spec.model}
    os.environ["AGENTSOCIETY_LLM_API_KEY"] = resolved["api_key"]
    os.environ["AGENTSOCIETY_LLM_API_BASE"] = resolved["api_base"]
    os.environ["AGENTSOCIETY_LLM_MODEL"] = resolved["model"]
    return {"provider": provider, "api_base": resolved["api_base"], "model": resolved["model"]}
