from dataclasses import dataclass

import networkx as nx

from inference import greedy_intervene, get_graph_fake_probability
from schemas import AnalyzeRequest, AnalyzeResponse
from services import BackendResources
from utils import bfs_depths


@dataclass
class DetectionResult:
    prediction: str
    confidence: float
    graph_fake_probability: float
    text_fake_probability: float
    model_status: str
    evidence: list[str]


class MisinformationPipeline:
    def __init__(self, resources: BackendResources) -> None:
        self.resources = resources

    def health_payload(self) -> dict:
        model = self.resources.get_model()
        inventory = self.resources.graph_inventory()
        return {
            "status": "ok",
            "service": "misinformation_pipeline",
            "version": "4.0.0",
            "model_status": model.status,
            "weights_loaded": model.weights_loaded,
            "graph_inventory": inventory,
        }

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        graph, root = self.resources.build_graph_from_input(request.text, request.graph_data)
        detection = self._run_detection(request.text or "", graph, root)
        causal_factors = self._run_causal_analysis(
            text=request.text or "",
            graph=graph,
            root=root,
            detection=detection,
        )
        intervention_result = self._run_intervention(
            graph=graph,
            root=root,
            prediction=detection.prediction,
            confidence=detection.confidence,
            k=request.k,
        )
        intervention_message = self._build_intervention_message(
            detection=detection,
            intervention_result=intervention_result,
        )

        return AnalyzeResponse(
            prediction=detection.prediction,
            confidence=round(detection.confidence, 4),
            causal_factors=causal_factors,
            intervention=intervention_message,
            graph_fake_probability=round(detection.graph_fake_probability, 4),
            intervention_nodes=intervention_result["intervention_nodes"],
            score_history=intervention_result["score_history"],
            reduction_pct=intervention_result["reduction_pct"],
            model_status=detection.model_status,
            graph_summary={
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "root": root,
                "depth": self._graph_depth(graph, root),
            },
        )

    def _run_detection(self, text: str, graph: nx.DiGraph, root: str) -> DetectionResult:
        model_handle = self.resources.get_model()
        graph_origin = graph.graph.get("origin", "unknown")
        graph_probability = get_graph_fake_probability(
            graph,
            root,
            model_handle.model,
            self.resources.device,
        )
        text_probability = self.resources.text_scorer.score(text) if text.strip() else graph_probability

        evidence = []
        heuristic_signals = self.resources.heuristic_signals(text)
        if heuristic_signals:
            evidence.extend(heuristic_signals)

        if graph_origin != "generated_from_text" and graph.number_of_nodes() >= 6 and graph.out_degree(root) >= 2:
            evidence.append("high_initial_cascade_branching")
        if graph_origin != "generated_from_text" and graph.number_of_edges() > graph.number_of_nodes():
            evidence.append("dense_propagation_pattern")

        if heuristic_signals:
            heuristic_boost = min(0.2, 0.06 * len(heuristic_signals))
            text_probability = min(0.99, text_probability + heuristic_boost)

        credible_signals = self.resources.credible_signals(text) if text.strip() else []
        if graph_origin == "generated_from_text":
            # Synthetic text graphs are scaffolding, not real propagation evidence.
            text_probability = 0.5 + ((text_probability - 0.5) * 0.25)
            if credible_signals and not heuristic_signals:
                credibility_discount = min(0.16, 0.03 * len(credible_signals))
                text_probability = max(0.05, text_probability - credibility_discount)

        if graph_origin == "generated_from_text":
            combined_probability = (0.9 * text_probability) + (0.1 * graph_probability)
        else:
            combined_probability = (0.6 * text_probability) + (0.4 * graph_probability)

        rumour_threshold = 0.62 if graph_origin == "generated_from_text" and not heuristic_signals else 0.5
        prediction = "rumour" if combined_probability >= rumour_threshold else "non-rumour"
        confidence = combined_probability if prediction == "rumour" else 1 - combined_probability

        if not evidence:
            if credible_signals:
                evidence.append("headline_matches_standard_news_style")
            else:
                evidence.append("no_strong_lexical_red_flags_detected")

        return DetectionResult(
            prediction=prediction,
            confidence=confidence,
            graph_fake_probability=graph_probability,
            text_fake_probability=text_probability,
            model_status=model_handle.status,
            evidence=evidence,
        )

    def _run_causal_analysis(
        self,
        text: str,
        graph: nx.DiGraph,
        root: str,
        detection: DetectionResult,
    ) -> list[str]:
        factors = list(detection.evidence)
        graph_origin = graph.graph.get("origin", "unknown")
        depth = self._graph_depth(graph, root)
        if graph_origin != "generated_from_text" and depth >= 4:
            factors.append("deep_cascade_structure")
        if graph_origin != "generated_from_text" and graph.number_of_nodes() >= 8:
            factors.append("wide_exposure_surface")
        if graph_origin != "generated_from_text" and detection.graph_fake_probability >= 0.65:
            factors.append("gnn_detected_high_graph_risk")
        if text and detection.text_fake_probability >= 0.65:
            factors.append("text_classifier_detected_misinformation_style")
        if detection.prediction == "non-rumour" and "gnn_detected_high_graph_risk" in factors:
            factors.append("content_looks_plausible_but_graph_structure_requires_monitoring")
        return factors[:6]

    def _run_intervention(
        self,
        graph: nx.DiGraph,
        root: str,
        prediction: str,
        confidence: float,
        k: int,
    ) -> dict:
        model_handle = self.resources.get_model()
        if graph.number_of_nodes() < 2 or graph.number_of_edges() == 0:
            return {
                "intervention_nodes": [],
                "score_history": [0.0],
                "reduction_pct": 0.0,
                "baseline_score": 0.0,
                "random_reduction": 0.0,
                "graph_fake_prob": round(confidence, 4),
            }

        result = greedy_intervene(
            graph,
            root,
            model_handle.model,
            self.resources.device,
            K=k,
        )

        if prediction == "non-rumour":
            result["intervention_nodes"] = []
            result["score_history"] = [result["baseline_score"]]
            result["reduction_pct"] = 0.0
        return result

    def _build_intervention_message(self, detection: DetectionResult, intervention_result: dict) -> str:
        nodes = intervention_result["intervention_nodes"]
        if detection.prediction == "rumour":
            if nodes:
                return (
                    "Throttle or fact-check the root claim first, then target cascade nodes "
                    f"{', '.join(nodes)} to reduce spread by about {intervention_result['reduction_pct']}%."
                )
            return "Flag the claim for manual review and publish a corrective notice before further amplification."
        return (
            "No aggressive containment needed. Keep the claim observable, retain provenance, "
            "and re-run intervention only if the cascade expands."
        )

    def _graph_depth(self, graph: nx.DiGraph, root: str) -> int:
        depths = bfs_depths(graph, root)
        return max(depths.values(), default=0)
