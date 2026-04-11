import re
import random
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import torch
import os
import networkx as nx
from pydantic import BaseModel

from model import RumourGAT
from utils import graph_to_json
from inference import greedy_intervene

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# =============================================================================
# NLP FAKE NEWS CLASSIFIER — trained in-memory at startup
# =============================================================================

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
    "sensex rises 300 points after rbi holds repo rate steady",
    "nifty gains 150 points on strong quarterly earnings",
    "rbi cuts repo rate by 25 basis points to boost growth",
    "rupee strengthens against dollar after trade surplus data",
    "india gdp grows at 6.5 percent in second quarter",
    "crude oil prices fall after opec increases production",
    # Politics
    "local elections scheduled for next month",
    "president signs infrastructure bill into law",
    "world leaders gather for climate summit in paris",
    "senate passes new budget resolution",
    "prime minister announces resignation after vote of no confidence",
    "congress approves foreign aid package",
    "supreme court rules on key environmental case",
    "un security council meets to discuss ceasefire",
    "parliament approves new defence budget for fiscal year",
    "government announces new policy on renewable energy",
    "modi inaugurates new metro line in delhi",
    "biden signs major climate bill into law",
    "trump signs executive order on immigration reform",
    "uk government announces new housing construction plan",
    "obama foundation launches new education initiative",
    # Science / Health
    "scientists discover new species of frog in amazon rainforest",
    "global temperatures show steady increase over past decade according to nasa",
    "new cancer treatment shows promising results in clinical trials",
    "researchers develop faster battery charging technology",
    "who reports decrease in malaria cases globally",
    "jwst captures image of distant galaxy cluster",
    "fda approves new diabetes medication",
    "study finds regular exercise reduces risk of heart disease",
    "fda approves new covid vaccine booster for adults",
    "nasa confirms successful mars rover landing",
    "who declares end of mpox public health emergency",
    "scientists publish study on vaccine safety in the lancet",
    "study finds no link between vaccines and autism",
    "researchers find new antibiotic to fight resistant bacteria",
    "isro successfully launches new earth observation satellite",
    "government releases official report on 5g network rollout",
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
    "amazon announces layoffs affecting thousands of employees",
    "microsoft acquires ai startup for two billion dollars",
    "electric vehicle sales hit record high this quarter",
    "india floods rescue teams deployed as dozens reported missing",
    "delhi air quality improves after new emission controls",
    "india signs trade agreement with gulf countries",
    "elon musk announces tesla quarterly earnings results",
    "putin meets with european leaders over ceasefire talks",
    "supreme court rules on social media content moderation case",
    "fact checkers debunk viral claim about water fluoridation",
    "world health organization updates covid guidance for member states",
    "rbi holds interest rates steady amid inflation concerns",
]

fake_news = [
    # Death hoaxes
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
    "shocking video shows world leaders are actually reptilian aliens",
    "government is poisoning water supply to control population",
    "chemtrails are spreading mind control chemicals worldwide",
    "new world order plans exposed insider reveals all",
    "bill gates plans to depopulate world using vaccines",
    "secret society controls all elections worldwide",
    "secret vaccine ingredient turns people into zombies leaked document",
    "government admits to faking moon landing in classified file",
    "deep state operatives planning false flag attack on capital",
    "reptilian shapeshifter caught on live television must watch video",
    "insider reveals truth about 5g towers and mind control agenda",
    "wake up sheeple the government is lying about everything again",
    "banned documentary exposes illuminati plan to control global food",
    # Miracle cures / scams
    "drink this miracle cure to never get sick again doctors hate it",
    "banned cancer cure big pharma does not want you to know",
    "this fruit extract cures diabetes in three days permanently",
    "ancient herb cures all diseases government is hiding it from you",
    "miracle herb cures cancer in 48 hours big pharma suppressed it",
    "you will not believe what they found inside the covid vaccine vials",
    # Urgent share viral
    "urgent share this before it gets deleted they are hiding it",
    "forward this to everyone you know before its too late",
    "local politician arrested in massive scam shocking details inside",
    "share this viral post to expose the truth about media lies",
    "breaking news you won't see on mainstream media exposed",
    "they don't want you to know this shocking truth revealed today",
    "scientist fired for revealing this cure they buried",
    "exposed elite plan to destroy economy and enslave population",
    "share now before this post gets removed truth about chemtrails",
    "must watch before removed shocking truth about government agenda",
    # Implausible financial claims
    "sensex crashes 9000 points in one day biggest collapse in history",
    "nifty rises 5000 points overnight after secret government deal",
    "rbi cuts interest rates by 40 percent in emergency move",
    "rupee falls 80 percent against dollar in single trading session",
    "india gdp grows 95 percent this quarter shocking economists",
]

corpus = real_news + fake_news
labels = [0] * len(real_news) + [1] * len(fake_news)

nlp_model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
    LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000),
)
nlp_model.fit(corpus, labels)
print(f"[OK] NLP Classifier trained — {len(real_news)} real + {len(fake_news)} fake samples")

# =============================================================================
# CONSTANTS
# =============================================================================

CREDIBLE_DOMAINS = [
    "reuters.com", "bbc.com", "apnews.com", "theguardian.com",
    "ndtv.com", "thehindu.com", "hindustantimes.com", "timesofindia.com",
    "bloomberg.com", "forbes.com", "wsj.com", "nytimes.com",
    "washingtonpost.com", "aljazeera.com", "indianexpress.com",
    "economictimes.indiatimes.com", "livemint.com", "businessstandard.com",
]

FAKE_PHRASES = [
    "you won't believe",
    "they don't want you to know",
    "share before deleted",
    "share before it gets deleted",
    "forward this to everyone",
    "forward this before",
    "banned cure",
    "deep state",
    "illuminati",
    "miracle cure",
    "doctors hate",
    "urgent share",
    "chemtrail",
    "reptilian",
    "new world order",
    "wake up sheeple",
    "secret agenda",
    "they are hiding",
    "big pharma suppressed",
    "must watch before removed",
    "shocking truth revealed",
    "share now before",
    "the government is lying to you",
    "mind control",
    "microchip injection",
]

# (trigger keywords, max plausible number, description)
PLAUSIBILITY_RULES = [
    (["sensex", "nifty"],                    2000, "index point move"),
    (["dow", "nasdaq", "s&p"],               3000, "index point move"),
    (["repo rate", "interest rate"],           15, "rate percent"),
    (["gdp"],                                  30, "gdp growth percent"),
    (["inflation"],                            50, "inflation percent"),
    (["rupee", "dollar", "currency"],          60, "currency move percent"),
]

# =============================================================================
# HELPERS
# =============================================================================

def check_numerical_plausibility(text: str) -> bool:
    """
    Returns True if the text contains a numerically implausible
    financial or statistical claim.

    Examples flagged:
        "Sensex rises 4400 points"     → True  (realistic max ~2000)
        "RBI cuts rate by 40 percent"  → True  (realistic max ~2%)
        "GDP grows 95 percent"         → True  (impossible)
    """
    numbers = []
    for raw in re.findall(r"[\d,]+\.?\d*", text):
        try:
            numbers.append(float(raw.replace(",", "")))
        except ValueError:
            continue

    if not numbers:
        return False

    for triggers, threshold, _ in PLAUSIBILITY_RULES:
        if any(t in text for t in triggers):
            if any(n > threshold for n in numbers):
                return True

    return False


def search_google_news(text: str):
    """
    Searches Google News RSS using first 6 words of the text.

    Returns:
        found        : bool — any results at all
        is_debunked  : bool — a fact-check/debunk article found
        result_count : int  — number of results
    """
    short_query = " ".join(text.split()[:6])
    query = urllib.parse.quote(short_query)
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urllib.request.urlopen(req, timeout=6) as response:
        xml_data = response.read()

    root_xml = ET.fromstring(xml_data)
    items = root_xml.findall(".//item")

    if not items:
        return False, False, 0

    debunk_keywords = [
        "fact check", "hoax", "fake", "debunked",
        "false claim", "misinformation", "misleading",
    ]
    is_debunked = any(
        kw in (item.find("title").text or "").lower()
        for item in items[:8]
        for kw in debunk_keywords
        if item.find("title") is not None
    )

    return True, is_debunked, len(items)


# =============================================================================
# FASTAPI APP
# =============================================================================

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


def generate_dummy_data():
    print("[WARNING] all_graphs.pkl not found — generating dummy data")
    dummy_graphs = []
    for i in range(10):
        G = nx.erdos_renyi_graph(n=12, p=0.3, directed=True)
        for node in G.nodes():
            G.nodes[node]["x"] = torch.randn(5).tolist()
        label = "rumour" if i % 2 == 0 else "non-rumour"
        dummy_graphs.append((G, 0, label))
    with open("all_graphs.pkl", "wb") as f:
        pickle.dump(dummy_graphs, f)
    return dummy_graphs


@app.on_event("startup")
def load_model():
    global model, all_graphs

    model = RumourGAT(in_channels=5, hidden=32, heads=4, dropout=0.3)
    if os.path.exists("rumour_gat.pt"):
        try:
            model.load_state_dict(torch.load("rumour_gat.pt", map_location=DEVICE))
            model.eval()
            print("[OK] Model weights loaded from rumour_gat.pt")
        except Exception as e:
            print(f"[ERROR] Could not load model weights: {e}")
    else:
        print("[WARNING] rumour_gat.pt not found — using random weights")

    if os.path.exists("all_graphs.pkl"):
        try:
            with open("all_graphs.pkl", "rb") as f:
                all_graphs = pickle.load(f)
            print(f"[OK] Loaded {len(all_graphs)} graphs from all_graphs.pkl")
        except Exception as e:
            print(f"[ERROR] Could not load all_graphs.pkl: {e}")
            all_graphs = generate_dummy_data()
    else:
        all_graphs = generate_dummy_data()


# =============================================================================
# ROUTES
# =============================================================================

@app.get("/")
def root():
    return {
        "status": "running",
        "cascades": len(all_graphs) if all_graphs else 0,
        "model": "RumourGAT",
    }


@app.get("/api/cascades")
def get_cascades():
    cascades = []
    for i, (G, root, label) in enumerate(all_graphs):
        if len(G.nodes) >= 3:
            cascades.append({
                "id": i,
                "label": label,
                "nodes": len(G.nodes),
                "edges": len(G.edges),
            })
        if len(cascades) >= 100:
            break
    return {"cascades": cascades}


class NewsRequest(BaseModel):
    news_text: str


@app.post("/api/score_news")
def score_news(request: NewsRequest):
    text = request.news_text.strip()

    # --- Reject: too short ---
    if len(text.split()) < 4:
        return {
            "status": "error",
            "message": "Input too short. Please enter a full news headline or claim.",
        }

    text_lower = text.lower()

    # --- Reject: does not look like news ---
    news_signals = [
        "president", "minister", "government", "police", "court", "market",
        "stock", "economy", "election", "hospital", "scientist", "research",
        "university", "company", "report", "announce", "launch", "sign",
        "approve", "discover", "rate", "percent", "billion", "million",
        "says", "confirms", "reveals", "claims", "nasa", "fda", "un",
        "senate", "study", "survey", "parliament", "military", "budget",
        "trade", "india", "china", "russia", "us", "uk", "delhi", "mumbai",
        "washington", "london", "beijing", "arrested", "killed", "dead",
        "dies", "injured", "rescue", "earthquake", "flood", "fire",
        "explosion", "summit", "treaty", "vaccine", "drug", "trial",
        "outbreak", "pandemic", "virus", "sensex", "nifty", "rbi", "isro",
        "gdp", "inflation", "rupee", "dollar", "crude", "oil",
    ]
    if not any(s in text_lower for s in news_signals):
        return {
            "status": "error",
            "message": "Cannot classify: Input does not appear to be a news headline or claim.",
        }

    # --- Step 1: ML Classification ---
    prob_rumour = nlp_model.predict_proba([text_lower])[0][1]
    print(f"[INFO] ML raw prob_rumour={prob_rumour:.4f} | text: {text[:60]}")

    # --- Step 2: Boost — obvious fake phrases ---
    if any(phrase in text_lower for phrase in FAKE_PHRASES):
        prob_rumour = min(prob_rumour + 0.25, 0.99)
        print(f"[INFO] Fake phrase detected — boosted to {prob_rumour:.4f}")

    # --- Step 3: Boost — numerically implausible claim ---
    if check_numerical_plausibility(text_lower):
        prob_rumour = min(prob_rumour + 0.35, 0.99)
        print(f"[INFO] Implausible number detected — boosted to {prob_rumour:.4f}")

    CONFIDENCE_THRESHOLD = 0.65

    if prob_rumour >= CONFIDENCE_THRESHOLD:
        label = "rumour"
        score = prob_rumour
        print(f"[INFO] ML confident FAKE ({prob_rumour:.4f}) — skipping web search")

    elif prob_rumour <= (1 - CONFIDENCE_THRESHOLD):
        label = "non-rumour"
        score = prob_rumour
        print(f"[INFO] ML confident REAL ({prob_rumour:.4f}) — skipping web search")

    else:
        # --- Step 4: ML uncertain → Google News fallback ---
        print(f"[INFO] ML uncertain ({prob_rumour:.4f}) — searching Google News...")
        try:
            found, is_debunked, result_count = search_google_news(text)

            if not found:
                print("[INFO] No news results — using ML fallback")
                label = "rumour" if prob_rumour >= 0.5 else "non-rumour"
                score = prob_rumour

            elif is_debunked:
                label = "rumour"
                score = 0.88
                print("[INFO] Debunk/fact-check found → FAKE")

            else:
                label = "non-rumour"
                score = 0.15
                print(f"[INFO] {result_count} legitimate articles found → REAL")

        except Exception as e:
            print(f"[WARNING] Google News search failed: {e} — using ML fallback")
            label = "rumour" if prob_rumour >= 0.5 else "non-rumour"
            score = prob_rumour

    # --- Map to a graph cascade of matching label ---
    matching_ids = [idx for idx, data in enumerate(all_graphs) if data[2] == label]
    mapped_cascade_id = random.choice(matching_ids) if matching_ids else 0

    return {
        "status": "success",
        "label": label,
        "score": round(score * 100, 1),
        "cascade_id": mapped_cascade_id,
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
        return {"error": "Graph too small to analyze"}

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
        "graph_fake_prob": result["graph_fake_prob"],
        "intervention_nodes": result["intervention_nodes"],
        "score_history": result["score_history"],
        "graph_before": graph_before,
        "graph_after": graph_after,
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

    label_counts = {}
    nodes, edges = [], []
    for G, _, label in all_graphs:
        label_counts[label] = label_counts.get(label, 0) + 1
        nodes.append(len(G.nodes))
        edges.append(len(G.edges))

    return {
        "total_cascades": len(all_graphs),
        "label_distribution": label_counts,
        "avg_nodes": round(sum(nodes) / len(nodes), 2),
        "avg_edges": round(sum(edges) / len(edges), 2),
    }