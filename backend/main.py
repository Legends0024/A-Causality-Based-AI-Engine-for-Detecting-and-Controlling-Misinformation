from pathlib import Path
import torch
torch.set_num_threads(1)

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from inference import greedy_intervene
from pipeline import MisinformationPipeline
from schemas import AnalyzeRequest
from services import BackendResources
from utils import graph_to_json


BASE_DIR = Path(__file__).resolve().parent
resources = BackendResources(BASE_DIR)
pipeline = MisinformationPipeline(resources)

app = FastAPI(
    title="Causality-Based AI Engine for Misinformation Control",
    description="End-to-end detection, causal analysis, and intervention API.",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "Request payload is invalid.",
            "details": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "The misinformation pipeline failed to process the request.",
            "details": str(exc),
        },
    )


@app.get("/")
def root() -> dict:
    return pipeline.health_payload()


@app.get("/api/health")
def health() -> dict:
    return pipeline.health_payload()


@app.get("/api/theory")
def get_theory() -> dict:
    import json
    import os
    theory_path = os.path.join(os.path.dirname(__file__), "theory.json")
    if os.path.exists(theory_path):
        with open(theory_path, "r") as f:
            return json.load(f)
    return {}


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    return pipeline.analyze(request).model_dump()


class ScoreNewsRequest(BaseModel):
    news_text: str


class LegacyAnalyzeRequest(BaseModel):
    cascade_id: int
    k: int = 5
    forced_label: str | None = None


def _get_cascade(cascade_id: int):
    graphs = resources.get_graphs()
    if cascade_id < 0 or cascade_id >= len(graphs):
        raise HTTPException(status_code=404, detail="Invalid cascade_id")
    return graphs[cascade_id]


def _pick_cascade_id(label: str) -> int:
    """Map 5-class verdict to the appropriate cascade type for visualization."""
    # Misinformation/Likely Misinformation/Uncertain → pick a 'rumour' cascade
    # Likely Credible/Credible → pick a 'non-rumour' cascade
    target = "rumour" if label in ["Misinformation", "Likely Misinformation", "Uncertain"] else "non-rumour"
    graphs = resources.get_graphs()
    for index, (_, _, graph_label) in enumerate(graphs):
        if graph_label == target:
            return index
    return 0


@app.get("/api/cascades")
def cascades() -> dict:
    payload = []
    for index, (graph, _root, label) in enumerate(resources.get_graphs()):
        payload.append(
            {
                "id": index,
                "label": label,
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
            }
        )
    return {"cascades": payload}


@app.get("/api/graph/{cascade_id}")
def graph(cascade_id: int) -> dict:
    graph_obj, root, label = _get_cascade(cascade_id)
    return {"label": label, **graph_to_json(graph_obj, root)}


@app.get("/api/stats")
def stats() -> dict:
    graphs = resources.get_graphs()
    if not graphs:
        raise HTTPException(status_code=503, detail="No graphs available")
    label_distribution: dict[str, int] = {}
    node_counts = []
    edge_counts = []
    for graph_obj, _root, label in graphs:
        label_distribution[label] = label_distribution.get(label, 0) + 1
        node_counts.append(graph_obj.number_of_nodes())
        edge_counts.append(graph_obj.number_of_edges())
    return {
        "total_cascades": len(graphs),
        "label_distribution": label_distribution,
        "avg_nodes": round(sum(node_counts) / len(node_counts), 2),
        "avg_edges": round(sum(edge_counts) / len(edge_counts), 2),
        "model_status": resources.get_model().status,
    }


@app.get("/api/world_headlines")
def world_headlines(limit: int = 12) -> dict:
    return resources.fetch_world_headlines(limit=limit)


@app.post("/api/score_news")
def score_news(request: ScoreNewsRequest) -> dict:
    """
    Multi-signal misinformation scoring engine.
    Implements the CBAE methodology (Section 3.4 of the paper):
    - Layer 1: Causal DAG + BERT classification (initial signal)
    - Layer 2: TF-IDF heuristic scoring (linguistic features)
    - Layer 3: NewsAPI cross-verification (external evidence)
    - Layer 4: Sensationalism & suspicious phrase detection
    Final verdict is a weighted fusion of all signals.
    """
    result = pipeline.analyze(AnalyzeRequest(text=request.news_text, k=3)).model_dump()
    news_lookup = resources.lookup_news(request.news_text, page_size=5)

    # ── Signal 1: BERT + Causal DAG prediction ──
    bert_label = result["prediction"]
    bert_confidence = result["confidence"]

    # ── Signal 2: TF-IDF heuristic score (trained on fake/real examples) ──
    text_fake_prob = resources.text_scorer.score(request.news_text)

    # ── Signal 3: Suspicious phrase & sensational claim detection ──
    suspicious_signals = resources.heuristic_signals(request.news_text)
    is_sensational = resources.is_sensational_unverified_claim(request.news_text)
    has_implausible = resources.has_implausible_number(request.news_text.lower())
    credible_signals = resources.credible_signals(request.news_text)

    # ── Signal 4: NewsAPI cross-verification ──
    newsapi_enabled = news_lookup.get("enabled", False)
    newsapi_error = news_lookup.get("error")
    total_results = news_lookup.get("total_results", 0)
    top_similarity = news_lookup.get("top_similarity", 0.0)

    # ── Weighted Fusion (as described in paper Section 3.4) ──
    # Start with TF-IDF probability as the base signal (it's actually trained)
    fake_score = text_fake_prob

    # Boost fake score for suspicious linguistic patterns
    if suspicious_signals:
        fake_score += 0.15 * len(suspicious_signals)
    if has_implausible:
        fake_score += 0.25

    # Sensational claims without strong evidence are likely fake
    if is_sensational:
        fake_score += 0.15

    # Reduce fake score for credible institutional language
    if credible_signals and not suspicious_signals:
        fake_score -= 0.10 * min(len(credible_signals), 3)

    # NewsAPI evidence adjustment (the most powerful signal)
    if newsapi_enabled and not newsapi_error:
        if top_similarity >= 0.5:
            # Strong match: a real news outlet reported the same thing
            fake_score -= 0.35
        elif top_similarity >= 0.3:
            # Partial match: some related coverage exists
            fake_score -= 0.15
        elif is_sensational and top_similarity < 0.2:
            # Dramatic claim but NO matching news coverage → strong fake signal
            fake_score += 0.20
        elif total_results == 0:
            # Zero results at all → suspicious
            fake_score += 0.15

    # Clamp to [0, 1]
    fake_score = max(0.0, min(1.0, fake_score))

    # ── Map to 5-class verdict (per paper Section 3.4) ──
    if fake_score >= 0.75:
        label = "Misinformation"
        score = round(fake_score * 100, 1)
    elif fake_score >= 0.55:
        label = "Likely Misinformation"
        score = round(fake_score * 100, 1)
    elif fake_score >= 0.45:
        label = "Uncertain"
        score = round(fake_score * 100, 1)
    elif fake_score >= 0.30:
        label = "Likely Credible"
        score = round((1 - fake_score) * 100, 1)
    else:
        label = "Credible"
        score = round((1 - fake_score) * 100, 1)

    # Build method trace for transparency
    method_parts = ["causality_ai_v5"]
    if suspicious_signals:
        method_parts.append("heuristic_flags")
    if is_sensational:
        method_parts.append("sensational_detector")
    if newsapi_enabled and not newsapi_error:
        if top_similarity >= 0.3:
            method_parts.append("newsapi_confirmed")
        elif is_sensational and top_similarity < 0.2:
            method_parts.append("newsapi_unverified")
        else:
            method_parts.append("newsapi_weak_match")

    method = "+".join(method_parts)
    cascade_id = _pick_cascade_id(label)

    return {
        "status": "success",
        "label": label,
        "score": score,
        "causal_factors": result["causal_factors"],
        "counterfactual_explanation": result["counterfactual_explanation"],
        "distribution": result["verdict_distribution"],
        "method": method,
        "cascade_id": cascade_id,
        "newsapi_query": news_lookup.get("query"),
        "sources": news_lookup.get("sources", []),
        "provider": news_lookup.get("provider", "newsapi"),
        "articles": news_lookup.get("articles", []),
        "newsapi_error": newsapi_error,
    }


@app.post("/api/analyze_legacy")
def analyze_legacy_alias(request: LegacyAnalyzeRequest) -> dict:
    return run_legacy_analysis(request)


@app.post("/api/analyze", include_in_schema=False)
def run_legacy_analysis(request: LegacyAnalyzeRequest) -> dict:
    graph_obj, root, label = _get_cascade(request.cascade_id)
    
    # Use forced label from scanner if provided to ensure consistency
    final_label = request.forced_label if request.forced_label else label
    
    result = greedy_intervene(
        graph_obj,
        root,
        resources.get_model().model,
        resources.device,
        K=request.k,
    )
    return {
        "cascade_id": request.cascade_id,
        "label": final_label,
        "dataset_label": label,
        "nodes": graph_obj.number_of_nodes(),
        "edges": graph_obj.number_of_edges(),
        "baseline_score": result["baseline_score"],
        "final_score": result["score_history"][-1],
        "reduction_pct": result["reduction_pct"],
        "random_reduction": result["random_reduction"],
        "graph_fake_prob": result["graph_fake_prob"],
        "intervention_nodes": result["intervention_nodes"],
        "score_history": result["score_history"],
        "graph_before": graph_to_json(graph_obj, root),
        "graph_after": graph_to_json(graph_obj, root, set(result["intervention_nodes"])),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
