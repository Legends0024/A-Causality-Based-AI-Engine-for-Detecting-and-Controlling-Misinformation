import random
import torch
from utils import extract_node_features


def build_graph_tensors(G, root, removed_nodes: set, device: torch.device):
    nodes_kept = [n for n in G.nodes if n not in removed_nodes]
    if len(nodes_kept) < 2:
        return None
    subgraph = G.subgraph(nodes_kept)
    if len(subgraph.edges) == 0:
        return None
    node_mapping = {node: idx for idx, node in enumerate(nodes_kept)}
    features = [extract_node_features(subgraph, node, root) for node in nodes_kept]
    x = torch.tensor(features, dtype=torch.float, device=device)
    edge_list = [
        [node_mapping[u], node_mapping[v]]
        for u, v in subgraph.edges
        if u in node_mapping and v in node_mapping
    ]
    if not edge_list:
        return None
    edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
    return x, edge_index, node_mapping, nodes_kept


def get_spread_score(G, root, model, device, removed_nodes: set = None) -> float:
    if removed_nodes is None:
        removed_nodes = set()
    result = build_graph_tensors(G, root, removed_nodes, device)
    if result is None:
        return 0.0
    x, edge_index, _, _ = result
    with torch.no_grad():
        model.eval()
        node_probs = model.node_scores(x, edge_index)
    return float(node_probs.sum().cpu())


def get_graph_fake_probability(G, root, model, device) -> float:
    result = build_graph_tensors(G, root, set(), device)
    if result is None:
        return 0.5
    x, edge_index, node_mapping, _ = result
    root_index = node_mapping.get(root, 0)
    with torch.no_grad():
        model.eval()
        prob = model.predict_proba(x, edge_index, root_index=root_index)
    return float(prob)


def greedy_intervene(G, root, model, device, K: int = 5) -> dict:
    K = min(max(K, 1), 10)
    baseline_score  = get_spread_score(G, root, model, device)
    graph_fake_prob = get_graph_fake_probability(G, root, model, device)
    removed       = set()
    score_history = [baseline_score]
    candidates    = [n for n in G.nodes if n != root]

    for _ in range(K):
        remaining = [n for n in candidates if n not in removed]
        if not remaining:
            break
        best_node  = None
        best_score = float("inf")
        for node in remaining:
            score = get_spread_score(G, root, model, device, removed | {node})
            if score < best_score:
                best_score = score
                best_node  = node
        if best_node is None:
            break
        removed.add(best_node)
        score_history.append(best_score)

    final_score   = score_history[-1]
    reduction_pct = (baseline_score - final_score) / baseline_score * 100 if baseline_score > 0 else 0.0

    random_nodes     = random.sample(candidates, min(K, len(candidates)))
    random_score     = get_spread_score(G, root, model, device, set(random_nodes))
    random_reduction = (baseline_score - random_score) / baseline_score * 100 if baseline_score > 0 else 0.0

    return {
        "intervention_nodes": [str(n) for n in removed],
        "score_history":      [round(s, 4) for s in score_history],
        "reduction_pct":      round(reduction_pct, 2),
        "baseline_score":     round(baseline_score, 4),
        "random_reduction":   round(random_reduction, 2),
        "graph_fake_prob":    round(graph_fake_prob, 4),
    }
