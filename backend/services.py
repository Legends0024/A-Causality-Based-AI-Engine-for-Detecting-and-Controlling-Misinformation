import os
import pickle
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import httpx
import networkx as nx
import torch
from dotenv import load_dotenv
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
    "pm narendra modi addresses parliament on social justice and reservation",
    "x community notes flags misleading political claims during election season",
    "supporters criticize government post on x after leader praises reformer",
    "news report says opposition leaders troll minister after public statement",
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

CREDIBLE_NEWS_SIGNALS = [
    "prime minister",
    "pm ",
    "narendra modi",
    "parliament",
    "minister",
    "government",
    "community notes",
    "x ",
    " x's",
    "court",
    "election",
    "official",
    "statement",
    "reported",
    "according to",
    "trade agreement",
    "policy",
]

NEWSAPI_COUNTRIES = ["in", "us", "gb", "au", "ca"]
NEWSAPI_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "on", "in", "to", "for", "and", "or", "but", "with", "by",
    "from", "at", "as", "after", "before", "into", "about", "his", "her",
    "their", "them", "this", "that", "these", "those", "he", "she", "it",
    "they", "pm", "says", "said",
}


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
        load_dotenv(self.base_dir / ".env")
        self.device = torch.device("cpu")
        self._model: ModelHandle | None = None
        self._graphs: list[tuple[nx.DiGraph, str, str]] | None = None
        self._graph_source = "uninitialized"
        self._model_lock = threading.Lock()
        self._graph_lock = threading.Lock()
        self.text_scorer = TextSignalScorer()
        self.newsapi_key = os.getenv("NEWSAPI_KEY", "").strip()

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
        graph.graph["origin"] = "dataset_cascade"
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
            graph.graph["origin"] = "dummy_dataset"
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

    def newsapi_enabled(self) -> bool:
        return bool(self.newsapi_key)

    def extract_keywords(self, text: str, max_terms: int = 8) -> str:
        words = re.findall(r"[a-zA-Z0-9']+", text.lower())
        terms: list[str] = []
        for word in words:
            normalized = word.strip("'")
            if len(normalized) < 3 or normalized in NEWSAPI_STOPWORDS:
                continue
            if normalized in terms:
                continue
            terms.append(normalized)
            if len(terms) >= max_terms:
                break
        return " ".join(terms)

    def _newsapi_get(self, endpoint: str, params: dict) -> dict:
        if not self.newsapi_key:
            return {}
        url = f"https://newsapi.org/v2/{endpoint}"
        merged_params = {**params, "apiKey": self.newsapi_key}
        with httpx.Client(timeout=12.0) as client:
            response = client.get(url, params=merged_params)
            response.raise_for_status()
            return response.json()

    def _headline_similarity(self, lhs: str, rhs: str) -> float:
        left_words = set(re.findall(r"[a-zA-Z0-9]+", lhs.lower()))
        right_words = set(re.findall(r"[a-zA-Z0-9]+", rhs.lower()))
        if not left_words or not right_words:
            return 0.0
        return len(left_words & right_words) / len(left_words | right_words)

    def lookup_news(self, text: str, page_size: int = 5) -> dict:
        if not self.newsapi_enabled():
            return {
                "enabled": False,
                "provider": "newsapi",
                "query": "",
                "total_results": 0,
                "sources": [],
                "articles": [],
                "matched": False,
                "top_similarity": 0.0,
                "error": "NEWSAPI_KEY not configured",
            }

        query = self.extract_keywords(text)
        if not query:
            return {
                "enabled": True,
                "provider": "newsapi",
                "query": "",
                "total_results": 0,
                "sources": [],
                "articles": [],
                "matched": False,
                "top_similarity": 0.0,
                "error": "No searchable keywords found",
            }

        page_size = max(1, min(page_size, 10))

        def perform_search(search_query: str) -> dict:
            return self._newsapi_get(
                "everything",
                {
                    "q": search_query,
                    "language": "en",
                    "sortBy": "relevancy",
                    "pageSize": page_size,
                },
            )

        try:
            payload = perform_search(query)
            if not (payload.get("articles") or []):
                fallback_query = self.extract_keywords(text, max_terms=4)
                if fallback_query and fallback_query != query:
                    payload = perform_search(fallback_query)
                    query = fallback_query
        except Exception as exc:
            return {
                "enabled": True,
                "provider": "newsapi",
                "query": query,
                "total_results": 0,
                "sources": [],
                "articles": [],
                "matched": False,
                "top_similarity": 0.0,
                "error": str(exc),
            }

        normalized_articles = []
        for article in payload.get("articles", []) or []:
            title = (article.get("title") or "").strip()
            similarity = self._headline_similarity(text, title)
            normalized_articles.append(
                {
                    "title": title,
                    "description": (article.get("description") or "").strip(),
                    "url": (article.get("url") or "").strip(),
                    "source": (article.get("source") or {}).get("name") or "Unknown",
                    "published_at": (article.get("publishedAt") or "").strip(),
                    "similarity": round(similarity, 3),
                }
            )
        normalized_articles.sort(key=lambda item: item["similarity"], reverse=True)
        sources = []
        for article in normalized_articles:
            if article["source"] not in sources:
                sources.append(article["source"])

        top_similarity = normalized_articles[0]["similarity"] if normalized_articles else 0.0
        return {
            "enabled": True,
            "provider": "newsapi",
            "query": query,
            "total_results": payload.get("totalResults", 0) or 0,
            "sources": sources[:5],
            "articles": normalized_articles[: max(1, min(page_size, 10))],
            "matched": top_similarity >= 0.2 and len(normalized_articles) > 0,
            "top_similarity": top_similarity,
            "error": None,
        }

    def fetch_world_headlines(self, limit: int = 12) -> dict:
        if not self.newsapi_enabled():
            return {
                "status": "fallback",
                "provider": "sample_feed",
                "count": 5,
                "articles": self.sample_world_headlines()[:5],
                "error": "NEWSAPI_KEY not configured",
            }

        target_limit = max(1, min(limit, 25))
        articles: list[dict] = []
        seen_titles: set[str] = set()
        per_country = max(3, min(6, target_limit // max(1, len(NEWSAPI_COUNTRIES) // 2)))

        try:
            for country in NEWSAPI_COUNTRIES:
                payload = self._newsapi_get(
                    "top-headlines",
                    {
                        "country": country,
                        "pageSize": per_country,
                    },
                )
                for article in payload.get("articles", []) or []:
                    title = (article.get("title") or "").strip()
                    if not title:
                        continue
                    lowered = title.lower()
                    if lowered in seen_titles:
                        continue
                    seen_titles.add(lowered)
                    articles.append(
                        {
                            "title": title,
                            "description": (article.get("description") or "").strip(),
                            "url": (article.get("url") or "").strip(),
                            "source": (article.get("source") or {}).get("name") or country.upper(),
                            "published_at": (article.get("publishedAt") or "").strip(),
                            "tag": country.upper(),
                        }
                    )
                    if len(articles) >= target_limit:
                        break
                if len(articles) >= target_limit:
                    break
        except Exception as exc:
            return {
                "status": "fallback",
                "provider": "sample_feed",
                "count": len(self.sample_world_headlines()[:target_limit]),
                "articles": self.sample_world_headlines()[:target_limit],
                "error": str(exc),
            }

        if not articles:
            return {
                "status": "fallback",
                "provider": "sample_feed",
                "count": len(self.sample_world_headlines()[:target_limit]),
                "articles": self.sample_world_headlines()[:target_limit],
                "error": "NewsAPI returned no headlines",
            }

        return {
            "status": "ok",
            "provider": "newsapi",
            "count": len(articles),
            "articles": articles[:target_limit],
            "error": None,
        }

    def sample_world_headlines(self) -> list[dict]:
        return [
            {"title": "Federal reserve raises interest rates by 25 basis points", "source": "Sample Feed", "tag": "US"},
            {"title": "World leaders gather for climate summit in Paris", "source": "Sample Feed", "tag": "EU"},
            {"title": "India signs trade agreement with Gulf countries", "source": "Sample Feed", "tag": "IN"},
            {"title": "NASA confirms successful Mars rover landing milestone", "source": "Sample Feed", "tag": "SCI"},
            {"title": "Apple reports record quarterly revenue driven by iPhone sales", "source": "Sample Feed", "tag": "BIZ"},
        ]

    def heuristic_signals(self, text: str) -> list[str]:
        lowered = text.lower()
        signals = [phrase for phrase in SUSPICIOUS_PHRASES if phrase in lowered]
        if self.has_implausible_number(lowered):
            signals.append("implausible_numerical_claim")
        if re.search(r"\b(share|viral|exposed|shocking)\b", lowered):
            signals.append("viral_amplification_language")
        return signals

    def credible_signals(self, text: str) -> list[str]:
        lowered = f" {text.lower()} "
        signals = [phrase.strip() for phrase in CREDIBLE_NEWS_SIGNALS if phrase in lowered]
        if re.search(r"\b(pm|president|minister|government|court|parliament)\b", lowered):
            signals.append("institutional_headline_pattern")
        if re.search(r"\b(troll|criticize|hails|addresses|flags)\b", lowered):
            signals.append("standard_political_news_verb")
        unique_signals: list[str] = []
        for signal in signals:
            if signal not in unique_signals:
                unique_signals.append(signal)
        return unique_signals

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
            graph.graph["origin"] = "provided_graph"
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
        graph.graph["origin"] = "generated_from_text"
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
