from dataclasses import dataclass
import networkx as nx
import torch

from inference import greedy_intervene, get_graph_fake_probability
from schemas import AnalyzeRequest, AnalyzeResponse
from services import BackendResources
from utils import bfs_depths
from causal import CausalGraphBuilder, CounterfactualEngine


@dataclass
class DetectionResult:
    prediction: str
    confidence: float
    verdict_distribution: dict[str, float]
    graph_fake_probability: float
    text_fake_probability: float
    model_status: str
    evidence: list[str]
    causal_embedding: torch.Tensor


class MisinformationPipeline:
    def __init__(self, resources: BackendResources) -> None:
        self.resources = resources
        self.causal_builder = CausalGraphBuilder()

    def health_payload(self) -> dict:
        resources = self.resources
        model = resources.get_model()
        inventory = resources.graph_inventory()
        return {
            "status": "ok",
            "service": "causality_ai_engine",
            "architecture": "3-layer-causal-nlp",
            "version": "5.0.0",
            "model_status": model.status,
            "weights_loaded": model.weights_loaded,
            "newsapi_enabled": resources.newsapi_enabled(),
            "graph_inventory": inventory,
        }

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        # 1. Causal Graph Construction Module
        causal_dag = self.causal_builder.build_dag({"text": request.text})
        causal_embedding = self.causal_builder.get_causal_embedding(causal_dag)
        
        # 2. Deep Learning Classification Layer (BERT + Causal Augmentation)
        detection = self._run_detection(
            text=request.text or "",
            causal_embedding=causal_embedding
        )
        
        # 3. Counterfactual Intervention Engine
        cf_engine = CounterfactualEngine(causal_dag)
        cf_explanation = cf_engine.generate_explanation(detection.prediction)
        
        # Legacy/Intervention Logic (GNN based for spread control)
        graph, root = self.resources.build_graph_from_input(request.text, request.graph_data)
        intervention_result = self._run_intervention(
            graph=graph,
            root=root,
            prediction=detection.prediction,
            confidence=detection.confidence,
            k=request.k,
        )
        
        return AnalyzeResponse(
            prediction=detection.prediction,
            confidence=round(detection.confidence, 4),
            verdict_distribution=detection.verdict_distribution,
            causal_factors=detection.evidence,
            counterfactual_explanation=cf_explanation,
            intervention=self._build_intervention_message(detection, intervention_result),
            graph_fake_probability=round(detection.graph_fake_probability, 4),
            intervention_nodes=intervention_result["intervention_nodes"],
            score_history=intervention_result["score_history"],
            reduction_pct=intervention_result["reduction_pct"],
            model_status=detection.model_status,
            graph_summary={
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "root": root,
                "causal_nodes": causal_dag.number_of_nodes(),
                "causal_edges": causal_dag.number_of_edges(),
            },
        )

    def _run_detection(self, text: str, causal_embedding: torch.Tensor) -> DetectionResult:
        model_handle = self.resources.get_model()
        
        # BERT classification with causal embedding (No Grad for memory efficiency)
        with torch.no_grad():
            verdict = model_handle.causal_transformer.predict_verdict(
                text=text,
                causal_vector=causal_embedding,
                device=self.resources.device
            )
        
        # Heuristic/GNN signals for legacy compatibility
        # We can still use the text scorer as an auxiliary signal
        text_probability = self.resources.text_scorer.score(text) if text.strip() else 0.5
        
        evidence = self.resources.heuristic_signals(text)
        if not evidence:
            evidence = ["headline_matches_standard_news_style" if verdict["label"] == "Credible" else "lexical_patterns_suggest_distrust"]

        return DetectionResult(
            prediction=verdict["label"],
            confidence=verdict["confidence"],
            verdict_distribution=verdict["distribution"],
            graph_fake_probability=text_probability, # Mocking GNN score for now
            text_fake_probability=text_probability,
            model_status=model_handle.status,
            evidence=evidence,
            causal_embedding=causal_embedding
        )

    def _run_intervention(self, graph, root, prediction, confidence, k) -> dict:
        """
        Intervention Engine logic.
        Only runs the spread suppression optimizer (Greedy Intervention)
        if the content is classified as Misinformation.
        """
        # If news is likely credible, we do NOT want to stop its spread.
        if prediction in ["Credible", "Likely Credible", "Uncertain"]:
            return {
                "intervention_nodes": [], 
                "score_history": [0.0], 
                "reduction_pct": 0.0,
                "msg": "Content satisfies credibility threshold. No spread suppression required."
            }

        model_handle = self.resources.get_model()
        if graph.number_of_nodes() < 2:
            return {"intervention_nodes": [], "score_history": [0.0], "reduction_pct": 0.0}
            
        with torch.no_grad():
            result = greedy_intervene(
                graph, root, model_handle.model, self.resources.device, K=k
            )
        
        return result

    def _build_intervention_message(self, detection: DetectionResult, intervention_result: dict) -> str:
        nodes = intervention_result["intervention_nodes"]
        if detection.prediction in ["Misinformation", "Likely Misinformation"]:
            if nodes:
                return f"Throttle the root source and intervene at cascade nodes {', '.join(nodes)}. Potential reduction: {intervention_result['reduction_pct']}%."
            return "Flag as misinformation and notify publishing platforms for immediate review."
        return "Source appears credible. Continued monitoring of propagation is recommended but no active suppression required."
