from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .providers import configure_agentsociety


VALID_ROLES = {"resident", "firm", "government"}


def parse_roles(value: str) -> set[str]:
    roles = {item.strip() for item in value.split(",") if item.strip()}
    unknown = roles - VALID_ROLES
    if unknown:
        raise ValueError(f"Unknown LLM roles: {', '.join(sorted(unknown))}")
    return roles


def run_preflight(
    *,
    provider: str,
    llm_roles: set[str],
    key_env: str | None = None,
    api_base: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Validate local API wiring and custom modules without making a request."""
    provider_info = configure_agentsociety(
        provider,
        key_env=key_env,
        api_base=api_base,
        model=model,
    )
    parsed_url = urlparse(provider_info["api_base"])
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError(
            f"Invalid OpenAI-compatible API base: {provider_info['api_base']!r}"
        )

    package_root = Path(__file__).resolve().parent
    os.environ["WORKSPACE_PATH"] = str(package_root)
    from agentsociety2.registry import scan_and_register_custom_modules

    registration = scan_and_register_custom_modules(package_root)
    errors = list(registration.get("errors", [])) + list(
        registration.get("registration_errors", [])
    )
    if errors:
        raise RuntimeError(
            "AgentSociety custom module registration failed: "
            + "; ".join(str(error) for error in errors)
        )
    return {
        "status": "ready",
        "network_called": False,
        "provider": provider_info,
        "llm_roles": sorted(llm_roles),
        "registered_agents": [
            item["class_name"] for item in registration.get("agents", [])
        ],
        "registered_environments": [
            item["class_name"] for item in registration.get("envs", [])
        ],
        "credential_check": (
            "not_required"
            if provider == "offline"
            else (
                "present"
                if bool(os.environ.get("AGENTSOCIETY_LLM_API_KEY"))
                else "missing"
            )
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate AgentSociety/API configuration without an API request"
    )
    parser.add_argument(
        "--provider",
        choices=[
            "offline",
            "hkust",
            "deepseek",
            "dashscope",
            "moonshot",
            "openai",
            "custom",
        ],
        default="offline",
    )
    parser.add_argument(
        "--llm-roles",
        default="",
        help="Comma-separated: resident,firm,government",
    )
    parser.add_argument("--key-env", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--model", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_preflight(
        provider=args.provider,
        llm_roles=parse_roles(args.llm_roles),
        key_env=args.key_env,
        api_base=args.api_base,
        model=args.model,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
