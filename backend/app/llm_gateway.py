"""LiteLLM gateway with ordered key-pool failover, built-in response caching, and failure simulation."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.config import KeyEntry, Settings

logger = logging.getLogger(__name__)

VALID_SIMULATED_STATUSES = {429, 500, 502, 503}


class AllProvidersUnavailableError(Exception):
    """Raised when every configured API key fails with a retryable or upstream error."""


@dataclass
class GatewayResult:
    reply: str
    resolution_source: str  # "LIVE_INFERENCE" or "LLM_CACHE"
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: Optional[str]
    fallback_attempts: int


def parse_simulate_failure_header(header_value: Optional[str]) -> Dict[str, int]:
    """
    Parse the X-Simulate-Failure header into a mapping of key_id (or '*') -> HTTP status code.

    Example:
        "key_1=429, key_2=500" -> {"key_1": 429, "key_2": 500}
        "*=429" -> {"*": 429}
    """
    if not header_value:
        return {}
    rules: Dict[str, int] = {}
    for raw_pair in header_value.split(","):
        pair = raw_pair.strip()
        if not pair or "=" not in pair:
            continue
        target, raw_status = pair.split("=", 1)
        target = target.strip()
        raw_status = raw_status.strip()
        if not target or not raw_status.isdigit():
            continue
        status = int(raw_status)
        if status in VALID_SIMULATED_STATUSES:
            rules[target] = status
    return rules


def _extract_tokens(response: Any) -> Tuple[int, int]:
    """Extract per-request (input_tokens, output_tokens) from a LiteLLM response."""
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return 0, 0

    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None) or 0
        completion_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None) or 0

    return int(prompt_tokens), int(completion_tokens)


def _extract_reply_text(response: Any) -> str:
    """Extract assistant reply text from a LiteLLM completion response."""
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        return ""
    first = choices[0]
    message = getattr(first, "message", None) if not isinstance(first, dict) else first.get("message")
    if message is None:
        return ""
    content = getattr(message, "content", None) if not isinstance(message, dict) else message.get("content")
    return (content or "").strip()


def _is_cache_hit(response: Any) -> bool:
    """Detect whether LiteLLM served the response from its built-in cache."""
    hidden = getattr(response, "_hidden_params", None)
    if hidden is None and isinstance(response, dict):
        hidden = response.get("_hidden_params")
    if isinstance(hidden, dict):
        return bool(hidden.get("cache_hit"))
    return False


def _is_failover_error(exc: Exception) -> bool:
    """Return True if an upstream error should trigger failover to the next key."""
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and (status_code == 429 or 500 <= status_code <= 599):
        return True
    name = type(exc).__name__.lower()
    if any(
        token in name
        for token in (
            "ratelimit",
            "serviceunavailable",
            "internalserver",
            "badgateway",
            "timeout",
            "apiconnection",
            "quota",
            "overloaded",
            "apierror",
        )
    ):
        return True
    msg = str(exc).lower()
    if any(
        token in msg
        for token in (
            "429",
            "rate limit",
            "quota",
            "resource_exhausted",
            "500",
            "502",
            "503",
            "504",
            "temporarily unavailable",
            "overloaded",
        )
    ):
        return True
    return False


class SimulatedProviderError(Exception):
    """Internal exception raised when X-Simulate-Failure injects a failure for a key."""

    def __init__(self, key_id: str, status_code: int) -> None:
        self.key_id = key_id
        self.status_code = status_code
        super().__init__(f"Simulated HTTP {status_code} failure for {key_id}")


class LLMGateway:
    """Manages LiteLLM built-in response caching and automatic key-pool failover."""

    def __init__(self, settings: Settings, litellm_module: Any = None) -> None:
        self.settings = settings
        self.keys: List[KeyEntry] = list(settings.llm_api_keys)
        self.cache_enabled: bool = settings.cache_enabled
        self._litellm = litellm_module

        if self._litellm is None:
            import litellm  # type: ignore[import-not-found]

            self._litellm = litellm

        self._configure_litellm()

    def _configure_litellm(self) -> None:
        """Configure LiteLLM telemetry, retries, and built-in Cache."""
        # Disable automatic internal retries so failover moves immediately to the next key
        if hasattr(self._litellm, "num_retries"):
            self._litellm.num_retries = 0
        if hasattr(self._litellm, "drop_params"):
            self._litellm.drop_params = True

        if not self.cache_enabled:
            self._litellm.cache = None
            return

        cache_cls = getattr(self._litellm, "Cache", None)
        if cache_cls is None:
            try:
                from litellm.caching.caching import Cache as cache_cls  # type: ignore[import-not-found]
            except Exception:
                cache_cls = None

        if cache_cls is not None:
            if self.settings.cache_type == "disk":
                os.makedirs(self.settings.disk_cache_dir, exist_ok=True)
                self._litellm.cache = cache_cls(
                    type="disk",
                    disk_cache_dir=self.settings.disk_cache_dir,
                )
            else:
                self._litellm.cache = cache_cls(type="local")
            logger.info("LiteLLM cache initialized (type=%s)", self.settings.cache_type)

    async def complete(
        self,
        messages: List[Dict[str, str]],
        simulate_failure_header: Optional[str] = None,
    ) -> GatewayResult:
        """
        Execute an LLM completion with LiteLLM caching and automatic failover across the key pool.

        Cache key is LiteLLM's exact-match hash over (model, messages).
        When ENABLE_FAILURE_SIMULATION=true and X-Simulate-Failure is active, cache reads are
        bypassed for that single request so failover behavior can be demonstrated on demand.
        """
        if not self.keys:
            raise AllProvidersUnavailableError("No API keys configured")

        simulated_failures: Dict[str, int] = {}
        if self.settings.enable_failure_simulation and simulate_failure_header:
            simulated_failures = parse_simulate_failure_header(simulate_failure_header)

        bypass_cache = bool(simulated_failures)
        start = time.perf_counter()
        fallback_attempts = 0
        last_error: Optional[Exception] = None

        for entry in self.keys:
            # Check per-request simulated failure hook before calling the upstream provider
            simulated_status = simulated_failures.get("*") or simulated_failures.get(entry.key_id)
            if simulated_status is not None:
                fallback_attempts += 1
                last_error = SimulatedProviderError(entry.key_id, simulated_status)
                logger.warning(
                    "Simulated failure on %s (status=%s); failing over to next key",
                    entry.key_id,
                    simulated_status,
                )
                continue

            call_kwargs: Dict[str, Any] = {
                "model": entry.model,
                "messages": messages,
                "api_key": entry.api_key,
                "timeout": self.settings.request_timeout_s,
                "num_retries": 0,
            }
            if self.cache_enabled:
                if bypass_cache:
                    call_kwargs["caching"] = False
                    call_kwargs["cache"] = {"no-cache": True, "no-store": True}
                else:
                    call_kwargs["caching"] = True

            try:
                response = await self._litellm.acompletion(**call_kwargs)
                reply_text = _extract_reply_text(response)
                if not reply_text:
                    raise RuntimeError(f"Empty response from model {entry.model} on {entry.key_id}")

                elapsed_ms = max(0, int(round((time.perf_counter() - start) * 1000)))
                input_tokens, output_tokens = _extract_tokens(response)
                cache_hit = (not bypass_cache) and _is_cache_hit(response)

                if cache_hit:
                    return GatewayResult(
                        reply=reply_text,
                        resolution_source="LLM_CACHE",
                        model=entry.model,
                        latency_ms=elapsed_ms,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        key_id=None,
                        fallback_attempts=0,
                    )

                return GatewayResult(
                    reply=reply_text,
                    resolution_source="LIVE_INFERENCE",
                    model=entry.model,
                    latency_ms=elapsed_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    key_id=entry.key_id,
                    fallback_attempts=fallback_attempts,
                )

            except Exception as exc:
                last_error = exc
                if _is_failover_error(exc):
                    fallback_attempts += 1
                    logger.warning(
                        "Provider error on %s (%s: %s); attempting failover",
                        entry.key_id,
                        type(exc).__name__,
                        exc,
                    )
                    continue
                # Even on unexpected provider errors (e.g. auth/quota), try remaining keys before 503
                fallback_attempts += 1
                logger.error(
                    "Unexpected error on %s (%s: %s); attempting failover",
                    entry.key_id,
                    type(exc).__name__,
                    exc,
                )
                continue

        logger.error("All %d configured keys failed. Last error: %s", len(self.keys), last_error)
        raise AllProvidersUnavailableError("all providers unavailable")
