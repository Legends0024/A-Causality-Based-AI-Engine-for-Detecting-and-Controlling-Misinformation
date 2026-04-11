const RAW = import.meta.env.VITE_API_URL;
/** Empty string = same-origin (use Vite dev proxy to backend). */
const API_BASE = RAW === undefined || RAW === "" ? "" : String(RAW).replace(/\/$/, "");

function apiUrl(path) {
  const p = path.startsWith("/") ? path : `/${path}`;
  return API_BASE ? `${API_BASE}${p}` : p;
}

// Generic fetch wrapper with timeout and error handling
async function apiFetch(path, options = {}, timeoutMs = 15000) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(apiUrl(path), {
      ...options,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
    });

    clearTimeout(timeout);

    if (!res.ok) {
      const ct = res.headers.get("content-type") || "";
      if (ct.includes("application/json")) {
        const errBody = await res.json().catch(() => ({}));
        const detail = errBody.detail;
        const msg =
          typeof detail === "string"
            ? detail
            : Array.isArray(detail)
              ? detail.map((d) => d.msg || d).join("; ")
              : `HTTP ${res.status}`;
        throw new Error(msg || `HTTP ${res.status}`);
      }
      const text = await res.text();
      const snippet = text.replace(/\s+/g, " ").trim().slice(0, 160);
      throw new Error(
        snippet
          ? `HTTP ${res.status}: ${snippet}`
          : `HTTP ${res.status} (backend unreachable or proxy misconfigured — check BACKEND_URL / VITE_API_URL)`
      );
    }

    return res.json();
  } catch (err) {
    clearTimeout(timeout);
    if (err.name === "AbortError") throw new Error("Request timed out. Is the backend running?");
    throw err;
  }
}

export const fetchCascades = () => apiFetch("/api/cascades");
export const fetchStats = () => apiFetch("/api/stats");
export const fetchHealth = () => apiFetch("/api/health");
export const fetchGraph = (id) => apiFetch(`/api/graph/${id}`);
export const fetchWorldHeadlines = (limit = 24) =>
  apiFetch(`/api/world_headlines?limit=${encodeURIComponent(limit)}`, {}, 20000);

/** First BERT inference can be slow; allow extra time. */
export const scoreNews = (news_text) =>
  apiFetch(
    "/api/score_news",
    {
      method: "POST",
      body: JSON.stringify({ news_text }),
    },
    120000
  );

export const analyzeCascade = (cascade_id, k) =>
  apiFetch("/api/analyze", {
    method: "POST",
    body: JSON.stringify({ cascade_id, k }),
  }, 60000);
