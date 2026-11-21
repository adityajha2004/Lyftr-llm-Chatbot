"""FastAPI application exposing the exact Backend API contract."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings, load_settings
from app.database import Database
from app.llm_gateway import AllProvidersUnavailableError, LLMGateway
from app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ResponseMetadata,
    SessionMessagesResponse,
    StoredMessage,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def create_app(
    settings: Optional[Settings] = None,
    litellm_module: Any = None,
) -> FastAPI:
    """Application factory for production and test runs."""
    cfg = settings or load_settings()
    db = Database(cfg.database_url)
    gateway = LLMGateway(cfg, litellm_module=litellm_module)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        db.init_schema()
        logger.info(
            "Backend started: keys_configured=%d, cache_enabled=%s, failure_simulation=%s",
            len(cfg.llm_api_keys),
            cfg.cache_enabled,
            cfg.enable_failure_simulation,
        )
        yield
        db.close()

    app = FastAPI(
        title="Lyftr AI LLM Chatbot Gateway",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.settings = cfg
    app.state.db = db
    app.state.gateway = gateway

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server exception: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        if not db.ping():
            raise HTTPException(status_code=503, detail="database unavailable")
        if len(gateway.keys) == 0:
            raise HTTPException(status_code=503, detail="no API keys configured")
        return HealthResponse(
            status="ok",
            keys_configured=len(gateway.keys),
            cache_enabled=gateway.cache_enabled,
        )

    @app.post("/chat", response_model=ChatResponse)
    async def chat(
        payload: ChatRequest,
        x_simulate_failure: Optional[str] = Header(default=None, alias="X-Simulate-Failure"),
    ) -> ChatResponse:
        # Resolve effective system prompt (request override > stored session override > default)
        effective_system_prompt = (
            payload.system_prompt
            or db.get_session_system_prompt(payload.session_id)
            or cfg.default_system_prompt
        )

        # Load prior session history from SQLite
        history = db.get_session_messages(payload.session_id)

        # Build the exact message list sent to the LLM: system prompt + history + new message
        llm_messages: List[Dict[str, str]] = [
            {"role": "system", "content": effective_system_prompt}
        ]
        for msg in history:
            llm_messages.append({"role": msg["role"], "content": msg["content"]})
        llm_messages.append({"role": "user", "content": payload.message})

        try:
            result = await gateway.complete(
                messages=llm_messages,
                simulate_failure_header=x_simulate_failure,
            )
        except AllProvidersUnavailableError:
            raise HTTPException(status_code=503, detail="all providers unavailable")

        metadata = ResponseMetadata(
            resolution_source=result.resolution_source,  # type: ignore[arg-type]
            model=result.model,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            key_id=result.key_id,
            fallback_attempts=result.fallback_attempts,
        )

        # Persist user message and assistant reply ONLY after a successful completion
        db.save_turn(
            session_id=payload.session_id,
            user_message=payload.message,
            assistant_reply=result.reply,
            assistant_metadata=metadata.model_dump(),
            system_prompt=payload.system_prompt,
        )

        return ChatResponse(
            session_id=payload.session_id,
            reply=result.reply,
            metadata=metadata,
        )

    @app.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
    async def get_session_messages(session_id: str) -> SessionMessagesResponse:
        raw_messages = db.get_session_messages(session_id)
        if not raw_messages:
            raise HTTPException(status_code=404, detail="session not found")

        stored = [
            StoredMessage(
                role=m["role"],
                content=m["content"],
                metadata=ResponseMetadata(**m["metadata"]) if m.get("metadata") else None,
            )
            for m in raw_messages
        ]
        return SessionMessagesResponse(session_id=session_id, messages=stored)

    return app


app: Optional[FastAPI] = None if os.getenv("LYFTR_TEST_MODE") == "1" else create_app()
