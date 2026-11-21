const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"
).replace(/\/+$/, "");

export function generateSessionId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `s-${crypto.randomUUID().slice(0, 12)}`;
  }
  return `s-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export async function fetchHealth() {
  const res = await fetch(`${API_BASE_URL}/health`);
  if (!res.ok) {
    throw new Error(`Health check failed (${res.status})`);
  }
  return res.json();
}

export async function sendChatMessage({
  sessionId,
  message,
  systemPrompt,
  simulateFailure,
}) {
  const headers = {
    "Content-Type": "application/json",
  };
  if (simulateFailure) {
    headers["X-Simulate-Failure"] = simulateFailure;
  }

  const body = {
    session_id: sessionId,
    message,
  };
  if (systemPrompt && systemPrompt.trim()) {
    body.system_prompt = systemPrompt.trim();
  }

  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });

  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    if (data && data.detail) {
      detail =
        typeof data.detail === "string"
          ? data.detail
          : Array.isArray(data.detail)
            ? data.detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
            : JSON.stringify(data.detail);
    }
    const err = new Error(detail);
    err.status = response.status;
    throw err;
  }

  return data;
}
