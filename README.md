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

Backend (Web Service):
- Root directory: backend
- Build: pip install -r requirements.txt
- Start: uvicorn main:app --host 0.0.0.0 --port $PORT
- Python version: 3.11

Frontend (Static Site):
- Root directory: frontend
- Build: npm install && npm run build
- Publish: dist
- Environment variable: VITE_API_URL = <your backend render URL>