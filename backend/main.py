from pathlib import Path

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


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    return pipeline.analyze(request).model_dump()


class ScoreNewsRequest(BaseModel):
    news_text: str


class LegacyAnalyzeRequest(BaseModel):
    cascade_id: int
    k: int = 5


def _get_cascade(cascade_id: int):
    graphs = resources.get_graphs()
    if cascade_id < 0 or cascade_id >= len(graphs):
        raise HTTPException(status_code=404, detail="Invalid cascade_id")
    return graphs[cascade_id]


def _pick_cascade_id(label: str) -> int:
    graphs = resources.get_graphs()
    for index, (_, _, graph_label) in enumerate(graphs):
        if graph_label == label:
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
    result = pipeline.analyze(AnalyzeRequest(text=request.news_text, k=3)).model_dump()
    news_lookup = resources.lookup_news(request.news_text, page_size=5)
    label = result["prediction"]
    score = round(result["confidence"] * 100, 1)
    method = "pipeline_v4"

    newsapi_enabled = news_lookup.get("enabled", False)
    newsapi_matched = news_lookup.get("matched", False)
    top_similarity = news_lookup.get("top_similarity", 0.0)
    total_results = news_lookup.get("total_results", 0)
    newsapi_error = news_lookup.get("error")

    # ── Rule 1: No articles found on NewsAPI → mark as fake ──
    # If NewsAPI is live (no API error) and returned zero articles,
    # the headline has no real-world coverage — treat it as fake.
    if (
        newsapi_enabled
        and not newsapi_error
        and total_results == 0
        and label == "non-rumour"
    ):
        label = "rumour"
        score = max(score, 65.0)
        method = "pipeline_v4+newsapi_no_coverage"

    # ── Rule 2: Articles found with high similarity → confirm as real ──
    # Only flip rumour → non-rumour if similarity is very high (≥ 0.55)
    # and the model wasn't very confident it's fake (< 60%).
    elif (
        newsapi_enabled
        and newsapi_matched
        and top_similarity >= 0.6
        and label == "rumour"
        and score < 60
    ):
        label = "non-rumour"
        score = max(score, 72.0)
        method = "pipeline_v4+newsapi_match"

    cascade_id = _pick_cascade_id(label)
    return {
        "status": "success",
        "label": label,
        "score": score,
        "cascade_id": cascade_id,
        "method": method,
        "sources": news_lookup.get("sources", []),
        "query": news_lookup.get("query", ""),
        "provider": news_lookup.get("provider", "newsapi"),
        "total_results": total_results,
        "articles": news_lookup.get("articles", []),
        "newsapi_error": newsapi_error,
    }


@app.post("/api/analyze_legacy")
def analyze_legacy_alias(request: LegacyAnalyzeRequest) -> dict:
    return run_legacy_analysis(request)


@app.post("/api/analyze", include_in_schema=False)
def run_legacy_analysis(request: LegacyAnalyzeRequest) -> dict:
    graph_obj, root, label = _get_cascade(request.cascade_id)
    result = greedy_intervene(
        graph_obj,
        root,
        resources.get_model().model,
        resources.device,
        K=request.k,
    )
    return {
        "cascade_id": request.cascade_id,
        "label": label,
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
