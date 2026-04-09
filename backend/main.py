from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import torch
from model import RumourGAT
from utils import graph_to_json
from inference import greedy_intervene

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cpu")
model = None
all_graphs = None

@app.on_event("startup")
def load_model():
    global model, all_graphs
    model = RumourGAT(in_channels=5, hidden=32, heads=4, dropout=0.3)
    model.load_state_dict(torch.load("rumour_gat.pt", map_location=DEVICE))
    model.eval()
    with open("all_graphs.pkl", "rb") as f:
        all_graphs = pickle.load(f)

@app.get("/")
def root():
    return {"status": "running", "cascades": len(all_graphs), "model": "RumourGAT"}

@app.get("/api/cascades")
def get_cascades():
    cascades = []
    for i, (G, root, label) in enumerate(all_graphs):
        if len(G.nodes) >= 5:
            cascades.append({
                "id": i,
                "label": label,
                "nodes": len(G.nodes),
                "edges": len(G.edges)
            })
        if len(cascades) >= 100:
            break
    return {"cascades": cascades}

from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    cascade_id: int
    k: int = 5

@app.post("/api/analyze")
def analyze(request: AnalyzeRequest):
    cascade_id = request.cascade_id
    k = min(max(request.k, 1), 10)
    if cascade_id < 0 or cascade_id >= len(all_graphs):
        return {"error": "Invalid cascade_id"}
    G, root, label = all_graphs[cascade_id]
    if len(G.nodes) < 3:
        return {"error": "Graph too small"}
    result = greedy_intervene(G, root, model, DEVICE, k)
    graph_before = graph_to_json(G, root)
    graph_after = graph_to_json(G, root, set(result["intervention_nodes"]))
    return {
        "cascade_id": cascade_id,
        "label": label,
        "nodes": len(G.nodes),
        "edges": len(G.edges),
        "baseline_score": result["baseline_score"],
        "final_score": result["score_history"][-1],
        "reduction_pct": result["reduction_pct"],
        "random_reduction": result["random_reduction"],
        "intervention_nodes": result["intervention_nodes"],
        "score_history": result["score_history"],
        "graph_before": graph_before,
        "graph_after": graph_after
    }

@app.get("/api/graph/{cascade_id}")
def get_graph(cascade_id: int):
    if cascade_id < 0 or cascade_id >= len(all_graphs):
        return {"error": "Invalid cascade_id"}
    G, root, label = all_graphs[cascade_id]
    graph = graph_to_json(G, root)
    return {"label": label, **graph}

@app.get("/api/stats")
def get_stats():
    total = len(all_graphs)
    labels = {}
    nodes = []
    edges = []
    for G, _, label in all_graphs:
        labels[label] = labels.get(label, 0) + 1
        nodes.append(len(G.nodes))
        edges.append(len(G.edges))
    avg_nodes = sum(nodes) / len(nodes) if nodes else 0
    avg_edges = sum(edges) / len(edges) if edges else 0
    return {
        "total_cascades": total,
        "label_distribution": labels,
        "avg_nodes": avg_nodes,
        "avg_edges": avg_edges
    }