"""
test.py — Full validation suite for the Causal Intervention Engine.

Run with:
    cd backend
    uv run python test.py
"""

import traceback
import random
import torch
import networkx as nx

from model import RumourGAT
from inference import (
    greedy_intervene,
    get_spread_score,
    get_graph_fake_probability,
    build_graph_tensors,
)
from utils import graph_to_json

DEVICE = torch.device("cpu")
PASS   = "  ✅ PASS"
FAIL   = "  ❌ FAIL"


def section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def check(condition: bool, label: str, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    print(f"{status}  {label}" + (f"  [{detail}]" if detail else ""))
    return condition


# ── 1. Model Instantiation ────────────────────────────────────────────────────
section("1. Model Instantiation")
try:
    model = RumourGAT(in_channels=5, hidden=32, heads=4, dropout=0.3)
    model.eval()
    check(True, "RumourGAT instantiated")
except Exception as e:
    check(False, f"RumourGAT instantiation failed: {e}")
    raise SystemExit(1)

# ── 2. Model Forward Pass ─────────────────────────────────────────────────────
section("2. Model Forward Pass — Shape & Range Checks")
NUM_NODES = 8
x_test = torch.randn(NUM_NODES, 5)
edges  = [[i, i + 1] for i in range(NUM_NODES - 1)]
ei     = torch.tensor(edges, dtype=torch.long).t().contiguous()

try:
    with torch.no_grad():
        logit = model(x_test, ei, root_index=0)
        check(logit.shape == torch.Size([1]) or logit.dim() == 0,
              "forward() returns scalar/[1] logit", f"shape={logit.shape}")

        prob = model.predict_proba(x_test, ei, root_index=0)
        check(0.0 <= float(prob) <= 1.0,
              "predict_proba() in [0, 1]", f"value={float(prob):.4f}")

        ns = model.node_scores(x_test, ei)
        check(ns.shape == torch.Size([NUM_NODES]),
              "node_scores() shape=[N]", f"shape={ns.shape}")
        check(bool((ns >= 0).all() and (ns <= 1).all()),
              "node_scores() all in [0, 1]", f"min={ns.min():.4f} max={ns.max():.4f}")
except Exception as e:
    check(False, f"Forward pass failed: {e}")
    traceback.print_exc()

# ── 3. Load Graphs ────────────────────────────────────────────────────────────
section("3. Load all_graphs.pkl")
import pickle
try:
    with open("all_graphs.pkl", "rb") as f:
        all_graphs = pickle.load(f)
    check(len(all_graphs) > 0, f"Loaded {len(all_graphs)} graphs")
except FileNotFoundError:
    check(False, "all_graphs.pkl NOT FOUND — run startup or generate dummy data")
    raise SystemExit(1)
except Exception as e:
    check(False, f"Load failed: {e}")
    raise SystemExit(1)

# ── 4. First Graph Sanity ─────────────────────────────────────────────────────
section("4. First Graph Sanity Checks")
G, root, label = all_graphs[0]
print(f"  Label={label}  Nodes={len(G.nodes)}  Edges={len(G.edges)}  Root={root}")
check(len(G.nodes) >= 2, "At least 2 nodes",      f"n={len(G.nodes)}")
check(len(G.edges) >= 1, "At least 1 edge",       f"e={len(G.edges)}")
check(root in G.nodes,   "Root exists in graph",  f"root={root}")
check(label in ("rumour", "non-rumour"), "Label valid", f"label={label}")

# ── 5. build_graph_tensors ────────────────────────────────────────────────────
section("5. build_graph_tensors()")
try:
    result = build_graph_tensors(G, root, set(), DEVICE)
    check(result is not None, "Returns a result")
    if result:
        x, edge_index, node_mapping, nodes_kept = result
        check(x.shape[1] == 5,        "Feature matrix has 5 cols",  f"shape={x.shape}")
        check(edge_index.shape[0] == 2, "edge_index shape=[2,E]",   f"shape={edge_index.shape}")
        check(root in node_mapping,   "Root in node_mapping")
        print(f"  nodes_kept={len(nodes_kept)}, edges={edge_index.shape[1]}")
except Exception as e:
    check(False, f"build_graph_tensors() failed: {e}")
    traceback.print_exc()

# ── 6. get_spread_score ───────────────────────────────────────────────────────
section("6. get_spread_score()")
try:
    s_full = get_spread_score(G, root, model, DEVICE)
    check(s_full >= 0.0, "Baseline score >= 0", f"score={s_full:.4f}")

    non_root = [n for n in G.nodes if n != root]
    if non_root:
        s_reduced = get_spread_score(G, root, model, DEVICE, {non_root[0]})
        check(s_reduced <= s_full, "Removing a node reduces score",
              f"{s_full:.4f} → {s_reduced:.4f}")

    s_empty = get_spread_score(G, root, model, DEVICE, set(G.nodes) - {root})
    check(s_empty == 0.0, "Removing all non-root nodes → 0.0", f"score={s_empty}")
except Exception as e:
    check(False, f"get_spread_score() failed: {e}")
    traceback.print_exc()

# ── 7. get_graph_fake_probability ─────────────────────────────────────────────
section("7. get_graph_fake_probability()")
try:
    fp = get_graph_fake_probability(G, root, model, DEVICE)
    check(0.0 <= fp <= 1.0, "Fake prob in [0, 1]", f"prob={fp:.4f}")
    print(f"  label='{label}' | model fake_prob={fp:.4f}")
except Exception as e:
    check(False, f"get_graph_fake_probability() failed: {e}")
    traceback.print_exc()

# ── 8. greedy_intervene ───────────────────────────────────────────────────────
section("8. greedy_intervene(K=5)")
try:
    result = greedy_intervene(G, root, model, DEVICE, K=5)

    required = {"intervention_nodes", "score_history", "reduction_pct",
                "baseline_score", "random_reduction", "graph_fake_prob"}
    check(required.issubset(result.keys()), "All required keys present")

    sh = result["score_history"]
    check(len(sh) >= 1, "score_history has entries",  f"len={len(sh)}")
    check(sh[0] == result["baseline_score"], "score_history[0] == baseline_score")
    check(all(sh[i] >= sh[i+1] for i in range(len(sh)-1)),
          "score_history non-increasing")
    check(str(root) not in result["intervention_nodes"],
          "Root NOT in intervention_nodes", f"root={root}")
    check(0.0 <= result["reduction_pct"] <= 100.0,
          "reduction_pct in [0, 100]", f"{result['reduction_pct']:.2f}%")
    check(result["reduction_pct"] >= result["random_reduction"],
          "Greedy beats random")

    print(f"\n  Baseline     : {result['baseline_score']:.4f}")
    print(f"  Final        : {result['score_history'][-1]:.4f}")
    print(f"  Reduction    : {result['reduction_pct']:.2f}%")
    print(f"  Random       : {result['random_reduction']:.2f}%")
    print(f"  Fake prob    : {result['graph_fake_prob']:.4f}")
    print(f"  Suppressed   : {result['intervention_nodes']}")
except Exception as e:
    check(False, f"greedy_intervene() failed: {e}")
    traceback.print_exc()

# ── 9. Numerical Plausibility Check ──────────────────────────────────────────
section("9. Numerical Plausibility Check")
from main import check_numerical_plausibility

cases = [
    ("sensex rises 4400 points after rbi holds rate",    True),
    ("sensex rises 300 points after rbi holds rate",     False),
    ("rbi cuts repo rate by 40 percent in emergency",    True),
    ("rbi cuts repo rate by 25 basis points",            False),
    ("india gdp grows 95 percent this quarter",          True),
    ("india gdp grows at 6.5 percent in second quarter", False),
    ("rupee falls 80 percent against dollar",            True),
]
for text, expected in cases:
    result = check_numerical_plausibility(text)
    check(result == expected,
          f"{'IMPLAUSIBLE' if expected else 'PLAUSIBLE'}: {text[:55]}",
          f"got={result}")

# ── 10. Stress Test ───────────────────────────────────────────────────────────
section("10. Stress Test — All Graphs")
errors, skipped, processed = 0, 0, 0
for i, (Gi, ri, li) in enumerate(all_graphs):
    if len(Gi.nodes) < 3:
        skipped += 1
        continue
    try:
        s  = get_spread_score(Gi, ri, model, DEVICE)
        fp = get_graph_fake_probability(Gi, ri, model, DEVICE)
        assert 0.0 <= fp <= 1.0
        assert s >= 0.0
        processed += 1
    except Exception as e:
        print(f"  ❌ Graph {i} ({li}): {e}")
        errors += 1
check(errors == 0,
      f"{processed} graphs processed, {skipped} skipped (too small)",
      f"{errors} errors")

# ── 11. graph_to_json ─────────────────────────────────────────────────────────
section("11. graph_to_json()")
try:
    gj = graph_to_json(G, root)
    check("nodes" in gj and "edges" in gj, "Returns nodes and edges keys")
    check(all("id" in n and "depth" in n and "type" in n for n in gj["nodes"]),
          "All nodes have id, depth, type fields")
    check(all("source" in e and "target" in e for e in gj["edges"]),
          "All edges have source, target fields")
except Exception as e:
    check(False, f"graph_to_json() failed: {e}")
    traceback.print_exc()

# ── Summary ───────────────────────────────────────────────────────────────────
section("Done — fix any ❌ before deploying")
print()