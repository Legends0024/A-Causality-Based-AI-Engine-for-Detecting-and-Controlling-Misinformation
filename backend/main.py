from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import torch
import os
import networkx as nx
from pydantic import BaseModel
from typing import List

# Ensure these files exist in your directory
from model import RumourGAT
from utils import graph_to_json
from inference import greedy_intervene

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# ---------------------------------------------------------
# TRAIN LOCAL NLP FAKE NEWS CLASSIFIER IN-MEMORY
# ---------------------------------------------------------
real_news = [
    # Economics
    "economy grows by 2% in the second quarter",
    "unemployment rate falls to 4 percent this month",
    "stock market reaches all time high after tech earnings",
    "federal reserve raises interest rates by 25 basis points",
    "gdp growth slows in third quarter report",
    "inflation drops to lowest level in two years",
    "oil prices rise amid middle east tensions",
    "central bank holds interest rates steady",
    # Politics
    "local elections scheduled for next month",
    "president signs infrastructure bill into law",
    "world leaders gather for climate summit in paris",
    "senate passes new budget resolution",
    "prime minister announces resignation after vote of no confidence",
    "congress approves foreign aid package",
    "supreme court rules on key environmental case",
    "un security council meets to discuss ceasefire",
    # Science / Health
    "scientists discover new species of frog in amazon rainforest",
    "global temperatures show steady increase over past decade according to nasa",
    "new cancer treatment shows promising results in clinical trials",
    "researchers develop faster battery charging technology",
    "who reports decrease in malaria cases globally",
    "jwst captures image of distant galaxy cluster",
    "fda approves new diabetes medication",
    "study finds regular exercise reduces risk of heart disease",
    # Local / General
    "new school opening in downtown area next fall",
    "city council approves budget for new highway",
    "tech company announces new smartphone release date",
    "local sports team wins championship game",
    "university launches new computer science program",
    "municipal workers begin repair of downtown bridge",
    "hospital expands emergency department capacity",
    "airlines report record passenger numbers this summer",
    "police arrest suspect in downtown robbery case",
    "fire department responds to warehouse blaze",
    "school district announces new literacy initiative",
    "mayor proposes new public transit expansion plan",
]

fake_news = [
    # Death hoaxes about public figures
    "trump is dead confirmed sources say",
    "biden found dead in white house this morning",
    "elon musk killed in mysterious car accident last night",
    "obama secretly arrested by military tribunal",
    "modi poisoned by deep state operatives in delhi",
    "celebrity died in secret government cover up",
    # Conspiracy / Hoax
    "shocking secret the earth is actually flat and government lied",
    "viral hoax exposes deep state scam against citizens",
    "alien spaceship found hidden in military base area 51",
    "illuminati controls world governments secret leaked document proves it",
    "you won't believe what they are hiding in your vaccines",
    "5g networks cause mind control and microchip injection in humans",
    "banned truth exposed the government is lying to you about everything",
    "did you hear the viral conspiracy about the moon landing being faked",
    "shocking video shows world leaders are actually reptilian aliens",
    "government is poisoning water supply to control population",
    "chemtrails are spreading mind control chemicals worldwide",
    "new world order plans exposed insider reveals all",
    "bill gates plans to depopulate world using vaccines",
    "secret society controls all elections worldwide",
    # Miracle cures / scams
    "drink this miracle cure to never get sick again doctors hate it",
    "banned cancer cure big pharma does not want you to know",
    "this fruit extract cures diabetes in three days permanently",
    "ancient herb cures all diseases government is hiding it from you",
    # Urgent share viral
    "urgent share this before it gets deleted they are hiding it",
    "forward this to everyone you know before its too late",
    "local politician arrested in massive scam shocking details inside",
    "share this viral post to expose the truth about media lies",
    "breaking news you won't see on mainstream media exposed",
    "they don't want you to know this shocking truth revealed today",
    "scientist fired for revealing this cure they buried",
    "exposed elite plan to destroy economy and enslave population",
]

corpus = real_news + fake_news
labels = [0] * len(real_news) + [1] * len(fake_news)  # 0 = non-rumour, 1 = rumour

nlp_model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
    LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000)
)
nlp_model.fit(corpus, labels)
print(f"[OK] Local NLP Fake News Classifier Trained ({len(real_news)} real + {len(fake_news)} fake samples)")
# ---------------------------------------------------------

app = FastAPI()

# Enable CORS for frontend communication
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

def generate_dummy_data():
    """Creates a sample all_graphs.pkl if it doesn't exist."""
    print("[WARNING] all_graphs.pkl not found. Generating dummy data...")
    dummy_graphs = []
    for i in range(10):
        # Create a random directed graph
        G = nx.erdos_renyi_graph(n=12, p=0.3, directed=True)
        # Add the 5-channel features your RumourGAT model expects
        for node in G.nodes():
            G.nodes[node]['x'] = torch.randn(5).tolist()
        
        root_node = 0
        label = "rumour" if i % 2 == 0 else "non-rumour"
        dummy_graphs.append((G, root_node, label))
    
    with open("all_graphs.pkl", "wb") as f:
        pickle.dump(dummy_graphs, f)
    return dummy_graphs

@app.on_event("startup")
def load_model():
    global model, all_graphs
    
    # 1. Initialize and Load Model Weights
    model = RumourGAT(in_channels=5, hidden=32, heads=4, dropout=0.3)
    if os.path.exists("rumour_gat.pt"):
        try:
            model.load_state_dict(torch.load("rumour_gat.pt", map_location=DEVICE))
            model.eval()
            print("[OK] Model weights loaded from rumour_gat.pt")
        except Exception as e:
            print(f"[ERROR] Error loading model weights: {e}")
    else:
        print("[ERROR] rumour_gat.pt NOT FOUND. Model running with random weights.")

    # 2. Load or Generate Graph Data
    if os.path.exists("all_graphs.pkl"):
        try:
            with open("all_graphs.pkl", "rb") as f:
                all_graphs = pickle.load(f)
            print(f"[OK] Loaded {len(all_graphs)} graphs from all_graphs.pkl")
        except Exception as e:
            print(f"[ERROR] Error loading all_graphs.pkl: {e}")
            all_graphs = generate_dummy_data()
    else:
        all_graphs = generate_dummy_data()

@app.get("/")
def root():
    return {"status": "running", "cascades": len(all_graphs) if all_graphs else 0, "model": "RumourGAT"}

@app.get("/api/cascades")
def get_cascades():
    cascades = []
    for i, (G, root, label) in enumerate(all_graphs):
        if len(G.nodes) >= 3: # Lowered threshold for dummy data
            cascades.append({
                "id": i,
                "label": label,
                "nodes": len(G.nodes),
                "edges": len(G.edges)
            })
        if len(cascades) >= 100:
            break
    return {"cascades": cascades}

class NewsRequest(BaseModel):
    news_text: str

@app.post("/api/score_news")
def score_news(request: NewsRequest):
    text = request.news_text.strip()
    
    # Validation constraint: Reject meaningless / extremely short text (e.g. "hi")
    words = text.split()
    if len(words) < 4:
        return {
            "status": "error",
            "message": "Input rejected: Text is too short to be classified as a news source. Please provide a full sentence or news headline."
        }
        
    text_lower = text.lower()
    
    # --- Check 1: Does this look like news at all? ---
    # News text usually references people, places, events or actions
    news_signals = [
        "president", "minister", "government", "police", "court", "market", "stock",
        "economy", "election", "hospital", "scientist", "research", "university",
        "company", "report", "announce", "dies", "dead", "killed", "arrested",
        "launch", "sign", "approve", "discover", "rate", "percent", "billion",
        "million", "news", "says", "confirms", "reveals", "claims", "secret",
        "viral", "hoax", "conspiracy", "cure", "vaccine", "5g", "trump", "biden",
        "modi", "musk", "obama", "putin", "nasa", "who", "fda", "un", "senate"
    ]
    words_in_text = set(text_lower.split())
    signal_matches = sum(1 for s in news_signals if s in text_lower)
    
    if signal_matches == 0:
        return {
            "status": "error",
            "message": "Cannot classify: Input does not appear to be a news headline or claim. Please enter a real news statement."
        }
    
    # --- Check 2: ML Classification with Live Web Fallback ---
    prob_rumour = nlp_model.predict_proba([text_lower])[0][1]
    
    CONFIDENCE_THRESHOLD = 0.65
    
    if prob_rumour >= CONFIDENCE_THRESHOLD:
        # ML is confident it's fake
        label = "rumour"
        score = prob_rumour
    elif prob_rumour <= (1 - CONFIDENCE_THRESHOLD):
        # ML is confident it's real
        label = "non-rumour"
        score = prob_rumour
    else:
        # ML is uncertain -> search Google News to verify
        print(f"[INFO] ML uncertain ({prob_rumour:.2f}), searching Google News for: {text[:60]}")
        try:
            import urllib.request
            import urllib.parse
            import xml.etree.ElementTree as ET
            
            query = urllib.parse.quote(text)
            url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            
            with urllib.request.urlopen(req, timeout=6) as response:
                xml_data = response.read()
                root_xml = ET.fromstring(xml_data)
                items = root_xml.findall('.//item')
                
                if len(items) == 0:
                    # No news outlet is reporting this -> likely fake
                    label = "rumour"
                    score = 0.80
                    print(f"[INFO] No news results found -> marking as rumour")
                else:
                    # Check top results for debunking signals
                    is_debunked = False
                    for item in items[:5]:
                        title = item.find('title').text.lower() if item.find('title') is not None else ""
                        if any(xw in title for xw in ["fact check", "hoax", "fake", "debunked", "false claim", "misinformation"]):
                            is_debunked = True
                            break
                    
                    if is_debunked:
                        label = "rumour"
                        score = 0.88
                        print(f"[INFO] Fact-check/debunk found in news results -> marking as rumour")
                    else:
                        label = "non-rumour"
                        score = 0.18
                        print(f"[INFO] {len(items)} legitimate news articles found -> marking as real")
                        
        except Exception as e:
            print(f"[WARNING] Google News search failed: {e}. Using ML score as fallback.")
            # Use ML's best guess when both methods fail
            label = "rumour" if prob_rumour >= 0.5 else "non-rumour"
            score = prob_rumour
        
    # Introduce small random fuzzing for visualization aesthetics
    score = min(score + (len(text) % 10 / 100.0), 0.99)
    
    # Map to an existing Fixed Cascade instead of endlessly generating new ones
    import random
    
    # Find all fixed cascades that match the same label
    matching_ids = [idx for idx, data in enumerate(all_graphs) if data[2] == label]
    
    if matching_ids:
        # Randomly assign one of the fixed cascades of the correct type
        mapped_cascade_id = random.choice(matching_ids)
    else:
        # Fallback if somehow none exist
        mapped_cascade_id = 0
        
    return {
        "status": "success",
        "label": label,
        "score": round(score * 100, 1),
        "cascade_id": mapped_cascade_id
    }

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
    if not all_graphs:
        return {"error": "No data loaded"}
        
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
