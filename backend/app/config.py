"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class KeyEntry:
    """Represents a single API key and its target LiteLLM model in the key pool."""

    key_id: str
    api_key: str
    model: str


@dataclass(frozen=True)
class Settings:
    """Runtime settings parsed from environment variables."""

    llm_api_keys: List[KeyEntry] = field(default_factory=list)
    database_url: str = "sqlite:///./data/app.db"
    allowed_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000"])
    enable_failure_simulation: bool = False
    default_model: str = "gemini/gemini-2.0-flash"
    default_system_prompt: str = "You are a helpful, concise AI assistant."
    cache_enabled: bool = True
    cache_type: str = "local"  # "local" (in-memory) or "disk"
    disk_cache_dir: str = "./data/litellm_cache"
    request_timeout_s: float = 30.0


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_settings() -> Settings:
    """Load and parse Settings from process environment variables."""
    default_model = os.getenv("LLM_MODEL", "gemini/gemini-2.0-flash").strip() or "gemini/gemini-2.0-flash"
    raw_keys = _parse_csv(os.getenv("LLM_API_KEYS", ""))
    raw_models = _parse_csv(os.getenv("LLM_MODELS", ""))

    key_entries: List[KeyEntry] = []
    for idx, api_key in enumerate(raw_keys, start=1):
        model = raw_models[idx - 1] if idx - 1 < len(raw_models) else default_model
        key_entries.append(KeyEntry(key_id=f"key_{idx}", api_key=api_key, model=model))

    allowed_origins = _parse_csv(os.getenv("ALLOWED_ORIGINS", "http://localhost:3000"))
    if not allowed_origins:
        allowed_origins = ["http://localhost:3000"]

    database_url = os.getenv("DATABASE_URL", "sqlite:///./data/app.db").strip() or "sqlite:///./data/app.db"
    enable_failure_simulation = _parse_bool(os.getenv("ENABLE_FAILURE_SIMULATION"), default=False)
    default_system_prompt = os.getenv(
        "DEFAULT_SYSTEM_PROMPT",
        "You are a helpful, concise AI assistant.",
    ).strip() or "You are a helpful, concise AI assistant."

    cache_enabled = _parse_bool(os.getenv("CACHE_ENABLED"), default=True)
    cache_type = os.getenv("CACHE_TYPE", "local").strip().lower() or "local"
    if cache_type not in {"local", "disk"}:
        cache_type = "local"
    disk_cache_dir = os.getenv("DISK_CACHE_DIR", "./data/litellm_cache").strip() or "./data/litellm_cache"

    try:
        request_timeout_s = float(os.getenv("REQUEST_TIMEOUT_S", "30"))
    except ValueError:
        request_timeout_s = 30.0

    return Settings(
        llm_api_keys=key_entries,
        database_url=database_url,
        allowed_origins=allowed_origins,
        enable_failure_simulation=enable_failure_simulation,
        default_model=default_model,
        default_system_prompt=default_system_prompt,
        cache_enabled=cache_enabled,
        cache_type=cache_type,
        disk_cache_dir=disk_cache_dir,
        request_timeout_s=request_timeout_s,
    )
