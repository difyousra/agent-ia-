from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AgentConfig:
    """Runtime config for routing, legal API, and debug exposure."""

    debug_agent_routing: bool = False
    legal_search_enabled: bool = False
    legal_api_url: str = ""
    legal_api_key: str = ""
    legal_api_timeout: int = 20
    default_jurisdiction: str = "DZ"
    router_backend: str = "rules"  # rules | llm

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            debug_agent_routing=_env_bool("DEBUG_AGENT_ROUTING", False),
            legal_search_enabled=_env_bool("LEGAL_SEARCH_ENABLED", False),
            legal_api_url=(os.getenv("LEGAL_API_URL") or "").strip(),
            legal_api_key=(os.getenv("LEGAL_API_KEY") or "").strip(),
            legal_api_timeout=int(os.getenv("LEGAL_API_TIMEOUT", "20")),
            default_jurisdiction=(os.getenv("DEFAULT_JURISDICTION") or "DZ").strip().upper(),
            router_backend=(os.getenv("ROUTER_BACKEND") or "rules").strip().lower(),
        )
