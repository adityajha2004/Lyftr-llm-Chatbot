# Lyftr AI — Full-Stack LLM Chatbot with Key Failover & Response Caching

A full-stack multi-turn LLM chatbot built with **React (Vite)**, **Python (FastAPI)**, **LiteLLM**, and **SQLite**, featuring automatic API key failover, built-in response caching, per-request token tracking, and on-demand failure simulation.

## 1. Live URLs

- **Frontend URL:** `[Insert your deployed frontend URL, e.g. https://your-app.vercel.app]`
- **Backend URL:** `[Insert your deployed backend URL, e.g. https://your-api.onrender.com]`
- **GitHub Repository:** `[Insert your public GitHub repository URL]`
- **Demo Video:** `[Insert your 2–3 minute demo video URL]`

> **Deployment checklist:**
> 1. Deploy `/backend` (Docker runtime) to Render, Railway, Fly.io, or Hugging Face Spaces with a persistent volume mounted at `/data`, `DATABASE_URL=sqlite:////data/app.db`, `LLM_API_KEYS=<key1>,<key2>`, `ENABLE_FAILURE_SIMULATION=true`, and `ALLOWED_ORIGINS=<your-frontend-url>`.
> 2. Deploy `/frontend` to Vercel, Netlify, Render, or Cloudflare Pages with `VITE_API_BASE_URL=<your-backend-url>`.

## 2. Run locally: with Docker and without

### With Docker (recommended)

1. Copy `.env.example` to `.env` and populate `LLM_API_KEYS` with at least two real API keys:
   ```bash
   cp .env.example .env
   ```
2. Start both services:
   ```bash
   make up
   ```
   - Backend API: `http://localhost:8000` (`GET /health`)
   - Frontend UI: `http://localhost:3000`
3. Stop the stack:
   ```bash
   make down
   ```

### Without Docker

**Backend:**
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export LLM_API_KEYS="your-key-1,your-key-2"
export DATABASE_URL="sqlite:///./data/app.db"
export ALLOWED_ORIGINS="http://localhost:3000"
export ENABLE_FAILURE_SIMULATION="true"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

**Frontend:**
```bash
cd frontend
npm install
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

**Run backend tests:**
```bash
cd backend
pytest -v
# or without pytest:
python3 -m unittest discover -s tests -v
```

## 3. Design decisions: key fallback, caching, SQLite session storage

- **Key pool & deterministic auto-fallback (`backend/app/llm_gateway.py`):**
  - `LLM_API_KEYS` is parsed in exact order into `key_1`, `key_2`, ... (with optional per-key model overrides via `LLM_MODELS`, defaulting to `LLM_MODEL`).
  - Internal LiteLLM retries are disabled (`num_retries=0`) so that HTTP 429, quota exhaustion, timeouts, or 5xx errors fail over immediately to the next key in the pool rather than stalling on a dead key.
  - Every attempt is tracked (`fallback_attempts`), and the winning key's logical ID (`key_1`, `key_2`, ...) is returned without ever exposing raw API keys. If all keys fail, the API returns a clean `503 {"detail": "all providers unavailable"}` and persists nothing to SQLite.
  - When `ENABLE_FAILURE_SIMULATION=true` and `X-Simulate-Failure` is sent, simulated failures apply strictly to that single request (never poisoning subsequent requests) and bypass cache reads for that request so failover is reliably observable even for previously cached prompts.
- **LiteLLM built-in response caching:**
  - Uses LiteLLM's native [`Cache`](https://docs.litellm.ai/docs/caching/all_caches) (`litellm.cache = Cache(type="local")` by default, or `type="disk"` persisted under `/data/litellm_cache`).
  - Because `session_id` is excluded from the completion payload, LiteLLM's exact-match [local cache](https://docs.litellm.ai/docs/caching/local_caching) hashes `(model, messages)` where `messages` is `[system_prompt, ...history, new_message]`. Sending the same first message in a new session hits `LLM_CACHE` immediately with `key_id: null`, `fallback_attempts: 0`, identical reply text, and original token counts, detected via `response._hidden_params["cache_hit"]`.
  - The backend runs with `--workers 1` in Docker so all requests share the same in-memory cache process.
- **SQLite session storage (`backend/app/database.py`):**
  - Uses Python's standard-library `sqlite3` with WAL mode (`PRAGMA journal_mode=WAL`) and auto-creates the required `messages` table (plus an index on `(session_id, id)` and a `sessions` table for per-session `system_prompt` overrides) on startup.
  - User and assistant turns are committed together inside a single transaction *only after* the LLM call succeeds, guaranteeing that failed (`422` or `503`) requests never pollute conversation history.

## 4. Verification: how to see failover, cache hits and per-message token usage

1. **Health check:**
   ```bash
   curl -s http://localhost:8000/health | jq
   # {"status":"ok","keys_configured":2,"cache_enabled":true}
   ```
2. **Live inference vs. cross-session cache hit:**
   ```bash
   # First call -> LIVE_INFERENCE, key_id="key_1"
   curl -s -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"session_id":"s-1","message":"What is the capital of France?"}' | jq

   # Same first message in a new session -> LLM_CACHE, key_id=null, faster latency_ms
   curl -s -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"session_id":"s-2","message":"What is the capital of France?"}' | jq
   ```
3. **Multi-turn memory & per-request token usage:**
   ```bash
   curl -s -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"session_id":"s-1","message":"What river runs through it?"}' | jq

   curl -s http://localhost:8000/sessions/s-1/messages | jq
   ```
4. **Failover via `X-Simulate-Failure`:**
   ```bash
   # key_1 fails with 429 -> succeeds on key_2 with fallback_attempts=1
   curl -s -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -H "X-Simulate-Failure: key_1=429" \
     -d '{"session_id":"s-fail-1","message":"Hello after 429"}' | jq

   # All keys fail -> HTTP 503 {"detail":"all providers unavailable"}, nothing saved
   curl -i -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -H "X-Simulate-Failure: *=429" \
     -d '{"session_id":"s-fail-all","message":"Should fail"}'
   ```
5. **Validation errors (`422`):**
   ```bash
   curl -i -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"session_id":"s-bad","message":"   "}'
   ```

## 5. Setup Used: your editor and AI tools

- **AI Assistant:** Gemini (full-stack architecture, FastAPI/LiteLLM backend, React/Vite frontend, Dockerfiles, and automated contract test suite).
- **Editor / Runtime:** `[Add your editor, e.g. VS Code / Cursor]` with Docker Desktop, Python 3.11, and Node.js 20.

## 6. Bonus features (if any): what you added beyond the requirements and why

- **Per-session System Prompt Override (`data-testid="system-prompt-input"`):** Persisted per session in SQLite so follow-up turns inherit the custom system prompt while keeping the `messages` history endpoint clean of system messages.
- **Interactive Failover Test Hook Selector in UI:** Allows demonstrating `key_1=429`, `key_1=500`, and `*=429` directly from the browser during testing and demo recordings without leaving the UI.
- **Live Health Indicator & One-Click New Session:** Displays configured key count and cache status from `GET /health` in the header, and lets testers reset `session_id` in one click to verify cross-session `LLM_CACHE` hits.
- **Per-Key Model Mapping (`LLM_MODELS`):** Optional comma-separated model list allowing heterogeneous provider/model failover (e.g., Gemini primary failing over to OpenAI or Groq) with zero code changes.
