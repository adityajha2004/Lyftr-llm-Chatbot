import React, { useEffect, useRef, useState } from "react";
import { fetchHealth, generateSessionId, sendChatMessage } from "./api";

export default function App() {
  // Fresh session_id on every page load (Section 5 contract)
  const [sessionId, setSessionId] = useState(() => generateSessionId());
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [simulateFailure, setSimulateFailure] = useState("");
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    fetchHealth()
      .then((data) => setHealth(data))
      .catch(() => setHealth({ status: "unreachable" }));
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleNewSession = () => {
    setSessionId(generateSessionId());
    setMessages([]);
    setInput("");
  };

  const handleSend = async (event) => {
    if (event) event.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || loading) return;

    // Clear input immediately after sending (Section 5 contract)
    setInput("");
    const userMsg = {
      id: `u-${Date.now()}-${Math.random()}`,
      role: "user",
      content: trimmed,
    };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const data = await sendChatMessage({
        sessionId,
        message: trimmed,
        systemPrompt,
        simulateFailure,
      });

      const assistantMsg = {
        id: `a-${Date.now()}-${Math.random()}`,
        role: "assistant",
        content: data.reply,
        metadata: data.metadata,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const statusPrefix = err.status ? `Error ${err.status}: ` : "Error: ";
      const errorMsg = {
        id: `e-${Date.now()}-${Math.random()}`,
        role: "error",
        content: `${statusPrefix}${err.message || "Unable to reach backend"}`,
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand-group">
          <div className="brand-dot" />
          <div>
            <h1>Lyftr AI Gateway Chat</h1>
            <p className="subtitle">
              Session <code>{sessionId}</code>
              {health?.status === "ok" && (
                <span className="health-pill ok">
                  {health.keys_configured} keys &middot; cache{" "}
                  {health.cache_enabled ? "on" : "off"}
                </span>
              )}
              {health?.status === "unreachable" && (
                <span className="health-pill warn">backend offline</span>
              )}
            </p>
          </div>
        </div>
        <button
          type="button"
          className="secondary-btn"
          onClick={handleNewSession}
          disabled={loading}
        >
          New Session
        </button>
      </header>

      <section className="controls-bar">
        <div className="control-field grow">
          <label htmlFor="system-prompt">System Prompt (optional)</label>
          <input
            id="system-prompt"
            type="text"
            data-testid="system-prompt-input"
            placeholder="Override default system prompt for this session..."
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            disabled={loading}
          />
        </div>
        <div className="control-field">
          <label htmlFor="simulate-failure">Failover Test Hook</label>
          <select
            id="simulate-failure"
            value={simulateFailure}
            onChange={(e) => setSimulateFailure(e.target.value)}
            disabled={loading}
          >
            <option value="">Normal (No simulation)</option>
            <option value="key_1=429">Simulate key_1=429</option>
            <option value="key_1=500">Simulate key_1=500</option>
            <option value="*=429">Simulate *=429 (All fail &rarr; 503)</option>
          </select>
        </div>
      </section>

      <main className="chat-window" aria-live="polite">
        {messages.length === 0 && (
          <div className="empty-state">
            <h2>Start a multi-turn conversation</h2>
            <p>
              Every assistant reply displays its resolution source (
              <code>LIVE_INFERENCE</code> or <code>LLM_CACHE</code>), model,
              latency, and per-request token usage. Start a new session and send
              the same first prompt to observe a cache hit.
            </p>
          </div>
        )}

        {messages.map((msg) => {
          if (msg.role === "user") {
            return (
              <div key={msg.id} className="message-row user">
                <div className="bubble user-bubble">{msg.content}</div>
              </div>
            );
          }

          if (msg.role === "error") {
            return (
              <div key={msg.id} className="message-row error" role="alert">
                <div className="bubble error-bubble">{msg.content}</div>
              </div>
            );
          }

          const meta = msg.metadata || {};
          const isCache = meta.resolution_source === "LLM_CACHE";

          return (
            <div key={msg.id} className="message-row assistant">
              <div
                className="bubble assistant-bubble"
                data-testid="assistant-message"
              >
                <div className="reply-text">{msg.content}</div>

                <div className="meta-bar">
                  <span
                    className={`meta-badge source ${isCache ? "cache" : "live"}`}
                    data-testid="meta-source"
                  >
                    {meta.resolution_source}
                  </span>

                  <span className="meta-chip">
                    <span className="meta-label">Model</span>
                    <span data-testid="meta-model">{meta.model}</span>
                  </span>

                  <span className="meta-chip">
                    <span className="meta-label">Latency</span>
                    <span data-testid="meta-latency">{meta.latency_ms} ms</span>
                  </span>

                  <span className="meta-chip">
                    <span className="meta-label">In</span>
                    <span data-testid="meta-input-tokens">
                      {meta.input_tokens}
                    </span>
                  </span>

                  <span className="meta-chip">
                    <span className="meta-label">Out</span>
                    <span data-testid="meta-output-tokens">
                      {meta.output_tokens}
                    </span>
                  </span>

                  {meta.key_id && (
                    <span className="meta-chip">
                      <span className="meta-label">Key</span>
                      <span>{meta.key_id}</span>
                    </span>
                  )}

                  {typeof meta.fallback_attempts === "number" &&
                    meta.fallback_attempts > 0 && (
                      <span className="meta-badge failover">
                        {meta.fallback_attempts} failover
                        {meta.fallback_attempts > 1 ? "s" : ""}
                      </span>
                    )}
                </div>
              </div>
            </div>
          );
        })}

        {loading && (
          <div className="message-row assistant">
            <div className="bubble assistant-bubble loading-bubble">
              Thinking...
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </main>

      <form className="composer" onSubmit={handleSend}>
        <textarea
          data-testid="chat-input"
          rows={1}
          placeholder="Type your message... (Enter to send, Shift+Enter for newline)"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
          maxLength={4000}
        />
        <button
          type="submit"
          data-testid="send-button"
          className="primary-btn"
          disabled={loading || !input.trim()}
        >
          {loading ? "Sending..." : "Send"}
        </button>
      </form>
    </div>
  );
}
