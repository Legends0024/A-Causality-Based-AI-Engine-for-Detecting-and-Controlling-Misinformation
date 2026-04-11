import pickle
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from model import RumourGAT


REAL_NEWS_EXAMPLES = [
    "economy grows by 2 percent in the second quarter",
    "federal reserve raises interest rates by 25 basis points",
    "sensex rises 300 points after rbi holds repo rate steady",
    "world leaders gather for climate summit in paris",
    "fda approves new diabetes medication after trials",
    "nasa confirms successful mars rover landing",
    "india signs trade agreement with gulf countries",
    "apple reports record quarterly revenue driven by iphone sales",
]

FAKE_NEWS_EXAMPLES = [
    "trump is dead confirmed sources say breaking news",
    "alien spaceship found hidden in military base area 51 proof",
    "microchips found inside covid vaccine vials scientist exposes truth",
    "miracle herb cures cancer in 48 hours big pharma suppressed it",
    "urgent share this before it gets deleted they are hiding it",
    "sensex crashes 9000 points in one day biggest stock collapse in history",
    "india gdp grows 95 percent this quarter shocking economists worldwide",
    "government admits to faking moon landing in classified file leaked",
]

SUSPICIOUS_PHRASES = [
    "breaking news you will not see",
    "share this before it gets deleted",
    "secret leaked document",
    "deep state",
    "miracle cure",
    "must watch before removed",
    "they are hiding it",
    "confirmed sources say",
]


@dataclass
class ModelHandle:
    model: RumourGAT
    status: str
    weights_loaded: bool
    source: str


class TextSignalScorer:
    def __init__(self) -> None:
        self._pipeline = None
        self._lock = threading.Lock()

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        with self._lock:
            if self._pipeline is None:
                corpus = REAL_NEWS_EXAMPLES + FAKE_NEWS_EXAMPLES
                labels = [0] * len(REAL_NEWS_EXAMPLES) + [1] * len(FAKE_NEWS_EXAMPLES)
                self._pipeline = make_pipeline(
                    TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                    LogisticRegression(max_iter=400, random_state=42),
                )
                self._pipeline.fit(corpus, labels)
        return self._pipeline

    def score(self, text: str) -> float:
        text = text.strip()
        if not text:
            return 0.5
        pipeline = self._ensure_pipeline()
        probability = pipeline.predict_proba([text])[0][1]
        return float(probability)


class BackendResources:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.device = torch.device("cpu")
        self._model: ModelHandle | None = None
        self._graphs: list[tuple[nx.DiGraph, str, str]] | None = None
        self._graph_source = "uninitialized"
        self._model_lock = threading.Lock()
        self._graph_lock = threading.Lock()
        self.text_scorer = TextSignalScorer()

    @property
    def model_path(self) -> Path:
        return self.base_dir / "rumour_gat.pt"

    @property
    def graph_path(self) -> Path:
        return self.base_dir / "all_graphs.pkl"

    def get_model(self) -> ModelHandle:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            model = RumourGAT(in_channels=5, hidden=32, heads=4, dropout=0.3)
            status = "fallback_random_weights"
            weights_loaded = False
            source = str(self.model_path)
            if self.model_path.exists():
                try:
                    state = torch.load(self.model_path, map_location=self.device)
                    model.load_state_dict(state)
                    weights_loaded = True
                    status = "weights_loaded"
                except Exception as exc:
                    status = f"fallback_random_weights:{exc}"
            else:
                source = "generated_at_runtime"
            model.eval()
            self._model = ModelHandle(
                model=model,
                status=status,
                weights_loaded=weights_loaded,
                source=source,
            )
            return self._model

    def get_graphs(self) -> list[tuple[nx.DiGraph, str, str]]:
        if self._graphs is not None:
            return self._graphs
        with self._graph_lock:
            if self._graphs is not None:
                return self._graphs
            if self.graph_path.exists():
                try:
                    with self.graph_path.open("rb") as handle:
                        payload = pickle.load(handle)
                    self._graphs = [self._normalize_graph_tuple(item) for item in payload]
                    self._graph_source = "all_graphs.pkl"
                    return self._graphs
                except Exception:
                    pass
            self._graphs = self._generate_dummy_graphs()
            self._graph_source = "generated_dummy_graphs"
            return self._graphs

    def _normalize_graph_tuple(self, item) -> tuple[nx.DiGraph, str, str]:
        graph, root, label = item
        if not isinstance(graph, nx.DiGraph):
            graph = nx.DiGraph(graph)
        node_mapping = {node: str(node) for node in graph.nodes}
        graph = nx.relabel_nodes(graph, node_mapping, copy=True)
        root = str(root)
        if root not in graph.nodes:
            root = next(iter(graph.nodes), "claim_root")
            if root not in graph.nodes:
                graph.add_node(root)
        normalized_label = "rumour" if str(label).lower() == "rumour" else "non-rumour"
        return graph, str(root), normalized_label

    def _generate_dummy_graphs(self) -> list[tuple[nx.DiGraph, str, str]]:
        graphs: list[tuple[nx.DiGraph, str, str]] = []
        for index in range(6):
            graph = nx.DiGraph()
            root = "claim_root"
            graph.add_node(root)
            previous = root
            for hop in range(1, 6):
                node_id = f"n{index}_{hop}"
                graph.add_edge(previous, node_id)
                if hop % 2 == 0:
                    branch_id = f"n{index}_{hop}_branch"
                    graph.add_edge(previous, branch_id)
                previous = node_id
            label = "rumour" if index % 2 == 0 else "non-rumour"
            graphs.append((graph, root, label))
        return graphs

    def graph_inventory(self) -> dict:
        graphs = self.get_graphs()
        rumour_count = sum(1 for _, _, label in graphs if label == "rumour")
        return {
            "count": len(graphs),
            "source": self._graph_source,
            "rumour_graphs": rumour_count,
        }

    def heuristic_signals(self, text: str) -> list[str]:
        lowered = text.lower()
        signals = [phrase for phrase in SUSPICIOUS_PHRASES if phrase in lowered]
        if self.has_implausible_number(lowered):
            signals.append("implausible_numerical_claim")
        if re.search(r"\b(share|viral|exposed|shocking)\b", lowered):
            signals.append("viral_amplification_language")
        return signals

    def has_implausible_number(self, text: str) -> bool:
        checks = [
            (r"sensex rises (\d{4,}) points", 2500),
            (r"sensex crashes (\d{4,}) points", 4000),
            (r"gdp grows (\d{2,}) percent", 25),
            (r"bitcoin rises (\d{3,}) percent", 500),
            (r"cuts? .*? (\d{2,}) percent", 10),
            (r"falls? (\d{2,}) percent against", 25),
        ]
        for pattern, threshold in checks:
            match = re.search(pattern, text)
            if match and int(match.group(1)) > threshold:
                return True
        return False

    def build_graph_from_input(self, text: str | None, graph_data) -> tuple[nx.DiGraph, str]:
        if graph_data is not None and graph_data.nodes:
            graph = nx.DiGraph()
            for node in graph_data.nodes:
                graph.add_node(str(node.id), features=node.features, **node.metadata)
            for edge in graph_data.edges:
                if str(edge.source) not in graph.nodes:
                    graph.add_node(str(edge.source))
                if str(edge.target) not in graph.nodes:
                    graph.add_node(str(edge.target))
                graph.add_edge(str(edge.source), str(edge.target), weight=edge.weight)
            root = str(graph_data.root_id or graph_data.nodes[0].id)
            if root not in graph.nodes:
                graph.add_node(root)
            if graph.number_of_edges() == 0 and graph.number_of_nodes() == 1:
                graph.add_edge(root, f"{root}_support")
            return graph, root
        return self._build_text_graph(text or "")

    def _build_text_graph(self, text: str) -> tuple[nx.DiGraph, str]:
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        content_tokens: list[str] = []
        for token in tokens:
            if len(token) < 3:
                continue
            if token in content_tokens:
                continue
            content_tokens.append(token)
            if len(content_tokens) == 8:
                break

        root = "claim_root"
        graph = nx.DiGraph()
        graph.add_node(root)
        if not content_tokens:
            content_tokens = ["claim", "signal"]

        previous = root
        for index, token in enumerate(content_tokens, start=1):
            node_id = f"{token}_{index}"
            graph.add_node(node_id)
            graph.add_edge(previous, node_id)
            if index % 2 == 0:
                graph.add_edge(root, node_id)
            previous = node_id
        return graph, root
