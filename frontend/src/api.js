const API = import.meta.env.VITE_API_URL || "http://localhost:8000"

export const fetchCascades = () => fetch(`${API}/api/cascades`).then(r => r.json())
export const fetchStats    = () => fetch(`${API}/api/stats`).then(r => r.json())
export const analyzeCascade = (cascade_id, k) =>
  fetch(`${API}/api/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cascade_id, k })
  }).then(r => r.json())