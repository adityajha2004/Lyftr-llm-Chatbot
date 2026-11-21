"""End-to-end tests for FastAPI contract, LiteLLM caching, key failover, and SQLite persistence."""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List

os.environ.setdefault("LYFTR_TEST_MODE", "1")

from fastapi.testclient import TestClient

from app.config import KeyEntry, Settings
from app.main import create_app


class FakeLiteLLMCache:
    """In-memory exact-match cache mirroring LiteLLM's Cache(type='local') behavior."""

    def __init__(self, type: str = "local", **_kwargs: Any) -> None:
        self.type = type
        self.store: Dict[str, Any] = {}

    @staticmethod
    def _make_key(model: str, messages: List[Dict[str, str]]) -> str:
        return json.dumps({"model": model, "messages": messages}, sort_keys=True)


class FakeLiteLLM:
    """Deterministic async mock of the litellm SDK for unit & integration testing."""

    Cache = FakeLiteLLMCache

    def __init__(self, failing_api_keys: Dict[str, int] | None = None, live_delay_s: float = 0.015) -> None:
        self.cache: FakeLiteLLMCache | None = None
        self.num_retries = 0
        self.drop_params = True
        self.failing_api_keys = failing_api_keys or {}
        self.live_delay_s = live_delay_s
        self.live_calls: List[Dict[str, Any]] = []

    async def acompletion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        api_key: str,
        caching: bool = False,
        cache: Dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> Any:
        no_cache = bool(cache and cache.get("no-cache"))
        no_store = bool(cache and cache.get("no-store"))
        cache_key = FakeLiteLLMCache._make_key(model, messages)

        if caching and not no_cache and self.cache is not None and cache_key in self.cache.store:
            cached = self.cache.store[cache_key]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=cached["reply"]))],
                usage=SimpleNamespace(
                    prompt_tokens=cached["input_tokens"],
                    completion_tokens=cached["output_tokens"],
                ),
                _hidden_params={"cache_hit": True},
            )

        if api_key in self.failing_api_keys:
            status = self.failing_api_keys[api_key]
            err = RuntimeError(f"Upstream HTTP {status}")
            err.status_code = status  # type: ignore[attr-defined]
            raise err

        await asyncio.sleep(self.live_delay_s)
        self.live_calls.append({"model": model, "messages": messages, "api_key": api_key})

        last_user_msg = messages[-1]["content"]
        reply = f"Reply to: {last_user_msg} (turns={len(messages) - 1})"
        input_tokens = sum(max(1, len(m["content"].split())) for m in messages) * 3
        output_tokens = max(1, len(reply.split())) * 2

        if caching and not no_store and self.cache is not None:
            self.cache.store[cache_key] = {
                "reply": reply,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
            usage=SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens),
            _hidden_params={"cache_hit": False},
        )


def _make_client(
    *,
    enable_failure_simulation: bool = True,
    failing_api_keys: Dict[str, int] | None = None,
) -> tuple[TestClient, FakeLiteLLM]:
    fake_litellm = FakeLiteLLM(failing_api_keys=failing_api_keys)
    settings = Settings(
        llm_api_keys=[
            KeyEntry(key_id="key_1", api_key="real-key-1", model="gemini/gemini-2.0-flash"),
            KeyEntry(key_id="key_2", api_key="real-key-2", model="gemini/gemini-2.0-flash"),
        ],
        database_url="sqlite:///:memory:",
        allowed_origins=["http://localhost:3000"],
        enable_failure_simulation=enable_failure_simulation,
        cache_enabled=True,
        cache_type="local",
    )
    app = create_app(settings=settings, litellm_module=fake_litellm)
    return TestClient(app), fake_litellm


class TestBackendAPI(unittest.TestCase):
    def test_health_endpoint(self) -> None:
        client, _ = _make_client()
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            resp.json(),
            {"status": "ok", "keys_configured": 2, "cache_enabled": True},
        )

    def test_chat_live_and_cache_hit_across_sessions(self) -> None:
        client, fake = _make_client()

        r1 = client.post(
            "/chat",
            json={"session_id": "s-1", "message": "What is the capital of France?"},
        )
        self.assertEqual(r1.status_code, 200)
        d1 = r1.json()
        self.assertEqual(d1["session_id"], "s-1")
        self.assertEqual(d1["metadata"]["resolution_source"], "LIVE_INFERENCE")
        self.assertEqual(d1["metadata"]["model"], "gemini/gemini-2.0-flash")
        self.assertEqual(d1["metadata"]["key_id"], "key_1")
        self.assertEqual(d1["metadata"]["fallback_attempts"], 0)
        self.assertGreater(d1["metadata"]["input_tokens"], 0)
        self.assertGreater(d1["metadata"]["output_tokens"], 0)

        r2 = client.post(
            "/chat",
            json={"session_id": "s-2", "message": "What is the capital of France?"},
        )
        self.assertEqual(r2.status_code, 200)
        d2 = r2.json()
        self.assertEqual(d2["reply"], d1["reply"])
        self.assertEqual(d2["metadata"]["resolution_source"], "LLM_CACHE")
        self.assertIsNone(d2["metadata"]["key_id"])
        self.assertEqual(d2["metadata"]["fallback_attempts"], 0)
        self.assertEqual(d2["metadata"]["input_tokens"], d1["metadata"]["input_tokens"])
        self.assertEqual(d2["metadata"]["output_tokens"], d1["metadata"]["output_tokens"])
        self.assertLessEqual(d2["metadata"]["latency_ms"], d1["metadata"]["latency_ms"])
        self.assertEqual(len(fake.live_calls), 1)

        r3 = client.post(
            "/chat",
            json={"session_id": "s-3", "message": "What is the capital of Japan?"},
        )
        self.assertEqual(r3.status_code, 200)
        self.assertEqual(r3.json()["metadata"]["resolution_source"], "LIVE_INFERENCE")
        self.assertEqual(len(fake.live_calls), 2)

    def test_system_prompt_override_persists_per_session(self) -> None:
        client, fake = _make_client()

        r1 = client.post(
            "/chat",
            json={
                "session_id": "s-sys",
                "message": "Hi",
                "system_prompt": "Respond like a pirate.",
            },
        )
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(fake.live_calls[-1]["messages"][0]["content"], "Respond like a pirate.")

        # Follow-up turn without system_prompt keeps the session override
        r2 = client.post("/chat", json={"session_id": "s-sys", "message": "How are you?"})
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(fake.live_calls[-1]["messages"][0]["content"], "Respond like a pirate.")

        # Stored messages must NOT include the system prompt
        hist = client.get("/sessions/s-sys/messages").json()
        self.assertTrue(all(m["role"] != "system" for m in hist["messages"]))

    def test_multi_turn_history_and_session_endpoint(self) -> None:
        client, fake = _make_client()

        r1 = client.post("/chat", json={"session_id": "s-multi", "message": "Hello"})
        self.assertEqual(r1.status_code, 200)

        r2 = client.post("/chat", json={"session_id": "s-multi", "message": "Remember me?"})
        self.assertEqual(r2.status_code, 200)

        self.assertEqual(len(fake.live_calls[-1]["messages"]), 4)
        self.assertGreater(
            r2.json()["metadata"]["input_tokens"],
            r1.json()["metadata"]["input_tokens"],
        )

        hist = client.get("/sessions/s-multi/messages")
        self.assertEqual(hist.status_code, 200)
        body = hist.json()
        self.assertEqual(body["session_id"], "s-multi")
        self.assertEqual(len(body["messages"]), 4)
        self.assertEqual([m["role"] for m in body["messages"]], ["user", "assistant", "user", "assistant"])
        self.assertIsNone(body["messages"][0].get("metadata"))
        self.assertEqual(body["messages"][1]["metadata"]["resolution_source"], "LIVE_INFERENCE")

        missing = client.get("/sessions/does-not-exist/messages")
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json(), {"detail": "session not found"})

    def test_simulated_failover_header(self) -> None:
        client, _ = _make_client(enable_failure_simulation=True)

        r1 = client.post(
            "/chat",
            json={"session_id": "s-fail-1", "message": "Test 429 failover"},
            headers={"X-Simulate-Failure": "key_1=429"},
        )
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r1.json()["metadata"]["key_id"], "key_2")
        self.assertEqual(r1.json()["metadata"]["fallback_attempts"], 1)

        r2 = client.post(
            "/chat",
            json={"session_id": "s-fail-2", "message": "Test 500 failover"},
            headers={"X-Simulate-Failure": "key_1=500"},
        )
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.json()["metadata"]["key_id"], "key_2")
        self.assertEqual(r2.json()["metadata"]["fallback_attempts"], 1)

        r3 = client.post(
            "/chat",
            json={"session_id": "s-fail-all", "message": "Should fail completely"},
            headers={"X-Simulate-Failure": "*=429"},
        )
        self.assertEqual(r3.status_code, 503)
        self.assertEqual(r3.json(), {"detail": "all providers unavailable"})
        self.assertEqual(client.get("/sessions/s-fail-all/messages").status_code, 404)

    def test_simulation_header_ignored_when_disabled(self) -> None:
        client, _ = _make_client(enable_failure_simulation=False)
        resp = client.post(
            "/chat",
            json={"session_id": "s-nosim", "message": "Hello"},
            headers={"X-Simulate-Failure": "*=429"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["metadata"]["key_id"], "key_1")

    def test_real_provider_failover_on_429(self) -> None:
        client, _ = _make_client(failing_api_keys={"real-key-1": 429})
        resp = client.post(
            "/chat",
            json={"session_id": "s-real-failover", "message": "Keep working!"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["metadata"]["key_id"], "key_2")
        self.assertEqual(resp.json()["metadata"]["fallback_attempts"], 1)

    def test_validation_422_errors_not_persisted(self) -> None:
        client, _ = _make_client()

        r1 = client.post("/chat", json={"session_id": "s-bad", "message": "   "})
        self.assertEqual(r1.status_code, 422)
        self.assertIn("detail", r1.json())

        r2 = client.post("/chat", json={"session_id": "s-bad", "message": "a" * 4001})
        self.assertEqual(r2.status_code, 422)

        r3 = client.post("/chat", json={"message": "hello"})
        self.assertEqual(r3.status_code, 422)

        r4 = client.post("/chat", content="{not-json", headers={"Content-Type": "application/json"})
        self.assertEqual(r4.status_code, 422)

        self.assertEqual(client.get("/sessions/s-bad/messages").status_code, 404)


if __name__ == "__main__":
    unittest.main()
