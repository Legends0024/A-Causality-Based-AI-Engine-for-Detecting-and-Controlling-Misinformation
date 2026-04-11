import torch
from utils import extract_node_features

def get_spread_score(G, root, model, device, removed_nodes=set()):
    """
    Run the GAT model on the graph (with removed_nodes excluded)
    and return the SUM of all node spread probabilities.
    
    Lower score = better containment.
    
    Steps:
    1. Build subgraph excluding removed_nodes
    2. Extract node features using extract_node_features()
    3. Build edge_index tensor
    4. Run model in eval mode with torch.no_grad()
    5. Apply sigmoid to output
    6. Return float(probs.sum())
    
    Return 0.0 if graph has < 2 nodes or no edges.
    """
    nodes_to_keep = [n for n in G.nodes if n not in removed_nodes]
    if len(nodes_to_keep) < 2:
        return 0.0
    subgraph = G.subgraph(nodes_to_keep)
    if len(subgraph.edges) == 0:
        return 0.0
    features = []
    # Create mapping from original node ID to 0...N-1
    node_mapping = {}
    for idx, node in enumerate(nodes_to_keep):
        feat = extract_node_features(subgraph, node, root)
        features.append(feat)
        node_mapping[node] = idx
        
    x = torch.tensor(features, dtype=torch.float).to(device)
    
    edge_list = []
    for u, v in subgraph.edges:
        edge_list.append([node_mapping[u], node_mapping[v]])
        
    if not edge_list:
        return 0.0
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)
    
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        probs = torch.sigmoid(out)
        return float(probs.sum().cpu())

def greedy_intervene(G, root, model, device, K=5):
    """
    Greedy causal intervention optimizer.
    
    Algorithm:
    1. Compute baseline_score = get_spread_score(G, root, model, device)
    2. For step in range(K):
       a. For each candidate node (not root, not already removed):
          - Compute score if we remove it
       b. Select node with lowest resulting score
       c. Add to removed set
    3. Return:
       - intervention_nodes: list of chosen node IDs (as strings)
       - score_history: [baseline, after_step1, after_step2, ...]
       - reduction_pct: float, % reduction from baseline to final
       - baseline_score: float
    
    Also compute random_score: remove K random nodes, get score.
    Use this to show our method beats random.
    """
    baseline_score = get_spread_score(G, root, model, device)
    removed = set()
    score_history = [baseline_score]
    candidates = [n for n in G.nodes if n != root]
    for _ in range(K):
        best_node = None
        best_score = float('inf')
        for node in candidates:
            if node in removed:
                continue
            temp_removed = removed | {node}
            score = get_spread_score(G, root, model, device, temp_removed)
            if score < best_score:
                best_score = score
                best_node = node
        if best_node is None:
            break
        removed.add(best_node)
        score_history.append(best_score)
    final_score = score_history[-1]
    reduction_pct = (baseline_score - final_score) / baseline_score * 100 if baseline_score > 0 else 0
    # random
    import random
    random_nodes = random.sample(candidates, min(K, len(candidates)))
    random_score = get_spread_score(G, root, model, device, set(random_nodes))
    random_reduction = (baseline_score - random_score) / baseline_score * 100 if baseline_score > 0 else 0
    return {
        "intervention_nodes": [str(n) for n in removed],
        "score_history": score_history,
        "reduction_pct": reduction_pct,
        "baseline_score": baseline_score,
        "random_reduction": random_reduction
    }