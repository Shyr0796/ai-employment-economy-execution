from __future__ import annotations

import logging
import time
from typing import Any


_COST_WARNING_FRAGMENT = (
    "not in built-in cost map and no prefix/region variant matched; "
    "cache cost fields will default to 0"
)


class _DisabledCacheCostWarningFilter(logging.Filter):
    """Drop only LiteLLM's irrelevant cache-price warning."""

    def filter(self, record: logging.LogRecord) -> bool:
        return _COST_WARNING_FRAGMENT not in record.getMessage()


def normalize_created_timestamp(value: Any) -> tuple[int, str | None]:
    """Return LiteLLM's required integer timestamp and any non-numeric raw value."""
    if isinstance(value, bool):
        return int(time.time()), str(value)
    if isinstance(value, int):
        return value, None
    if isinstance(value, float):
        return int(value), None
    try:
        return int(float(value)), None
    except (TypeError, ValueError):
        return int(time.time()), None if value is None else str(value)


def install_openai_sdk_created_compat() -> bool:
    """Normalize a gateway string timestamp before OpenAI SDK serialization."""
    from openai.types.chat import ChatCompletion

    current = ChatCompletion.model_dump
    if bool(getattr(current, "_ai_economy_created_compat", False)):
        return False

    def model_dump(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        original = getattr(self, "created", None)
        normalized, raw = normalize_created_timestamp(original)
        if raw is None:
            return current(self, *args, **kwargs)

        self.created = normalized
        try:
            result = current(self, *args, **kwargs)
        finally:
            self.created = original
        result["provider_created_raw"] = raw
        return result

    setattr(model_dump, "_ai_economy_created_compat", True)
    ChatCompletion.model_dump = model_dump
    return True


def install_aiohttp_created_compat() -> bool:
    """Patch LiteLLM's aiohttp OpenAI adapter to normalize nonstandard `created`.

    HKUST returns a wall-clock string while LiteLLM's ModelResponse schema
    requires a Unix timestamp integer. The upstream httpx adapter already
    normalizes this field, but its aiohttp adapter currently does not.
    """
    from litellm.llms.aiohttp_openai.chat.transformation import (
        AiohttpOpenAIChatConfig,
    )

    current = AiohttpOpenAIChatConfig.transform_response
    if bool(getattr(current, "_ai_economy_created_compat", False)):
        return False

    async def transform_response(self: Any, *args: Any, **kwargs: Any) -> Any:
        response = await current(self, *args, **kwargs)
        normalized, raw = normalize_created_timestamp(
            getattr(response, "created", None)
        )
        response.created = normalized
        if raw is not None:
            hidden = getattr(response, "_hidden_params", None)
            if isinstance(hidden, dict):
                hidden["provider_created_raw"] = raw
        return response

    setattr(transform_response, "_ai_economy_created_compat", True)
    AiohttpOpenAIChatConfig.transform_response = transform_response
    return True


def install_disabled_cache_cost_warning_filter() -> bool:
    """Hide only cache-price noise when EconomicAgent disables caching."""
    logger = logging.getLogger("LiteLLM")
    if any(
        isinstance(item, _DisabledCacheCostWarningFilter)
        for item in logger.filters
    ):
        return False
    logger.addFilter(_DisabledCacheCostWarningFilter())
    return True


def install_litellm_compat() -> dict[str, bool]:
    """Install narrowly scoped compatibility fixes in the current worker."""
    return {
        "openai_sdk_created": install_openai_sdk_created_compat(),
        "aiohttp_created": install_aiohttp_created_compat(),
        "disabled_cache_cost_warning": (
            install_disabled_cache_cost_warning_filter()
        ),
    }
