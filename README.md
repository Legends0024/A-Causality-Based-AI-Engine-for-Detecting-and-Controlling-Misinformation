# Causal Intervention Engine for Misinformation Containment

This is a full-stack AI web application that uses a trained Graph Attention Network (GAT) to predict and optimize the containment of misinformation spread on social networks.

## Features

- GAT model for spread prediction
- Greedy causal intervention optimizer
- Before/after graph visualization
- React frontend with dark theme

## Setup

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

## Deployment

Backend (Web Service, e.g. Render — see `render.yaml`):
- Root directory: backend
- Build: pip install -r requirements.txt
- Start: uvicorn main:app --host 0.0.0.0 --port $PORT
- Python version: 3.11
- Set `NEWSAPI_KEY` in the host’s environment for live NewsAPI + headline feed.

Frontend (e.g. Vercel):
- Root directory: frontend
- Build: npm install && npm run build
- Publish: dist

**API routing when deployed (Vercel):** The app calls **relative** `/api/...` (leave `VITE_API_URL` unset). Add a Vercel environment variable **`BACKEND_URL`** = your live API origin (e.g. `https://your-service.onrender.com`, no trailing slash). The serverless handler `frontend/api/[...slug].js` proxies each `/api/*` request to that backend. Redeploy after setting or changing `BACKEND_URL`.

**Alternative:** Set **`VITE_API_URL`** at build time to the same backend origin and redeploy; the browser will call the API directly (CORS is already open on the FastAPI app).