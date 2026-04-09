import networkx as nx

def extract_node_features(G, node, root, max_depth=10):
    """
    Returns a 5-dimensional feature vector for a node:
    [depth_normalised, degree_normalised, in_degree_norm,
     out_degree_norm, is_root]
    
    Use nx.shortest_path() NOT nx.shortest_path_length()
    because newer NetworkX versions removed bidirectional_shortest_path.
    Also wrap in try/except — some nodes may not be reachable from root.
    """
    # depth
    try:
        path = nx.shortest_path(G, root, node)
        depth = len(path) - 1
    except nx.NetworkXNoPath:
        depth = -1
    depth_normalised = depth / max_depth if depth >= 0 else 0
    # degrees
    degree = G.degree(node)
    in_degree = G.in_degree(node)
    out_degree = G.out_degree(node)
    total_nodes = len(G.nodes)
    degree_normalised = degree / total_nodes if total_nodes > 0 else 0
    in_degree_norm = in_degree / total_nodes if total_nodes > 0 else 0
    out_degree_norm = out_degree / total_nodes if total_nodes > 0 else 0
    is_root = 1 if node == root else 0
    return [depth_normalised, degree_normalised, in_degree_norm, out_degree_norm, is_root]

def graph_to_json(G, root, intervention_nodes=None):
    """
    Convert a NetworkX DiGraph to a JSON-serialisable dict
    for the frontend D3/SVG renderer.
    
    Returns:
    {
      "nodes": [
        {
          "id": str(node),
          "depth": int,        # depth from root, -1 if unreachable
          "type": str,         # "root" | "debunked" | "infected"
          "degree": int
        },
        ...
      ],
      "edges": [
        {"source": str(u), "target": str(v)},
        ...
      ]
    }
    
    Node type logic:
    - if node == root        → "root"
    - if node in intervention_nodes → "debunked"
    - else                   → "infected"
    
    Use manual BFS (not nx.descendants or nx.bfs_edges) because
    newer NetworkX versions have removed these. BFS from root
    through G.successors(node).
    """
    if intervention_nodes is None:
        intervention_nodes = set()
    # BFS to compute depths
    depths = {}
    visited = set()
    queue = [root]
    depths[root] = 0
    visited.add(root)
    while queue:
        current = queue.pop(0)
        for neighbor in G.successors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                depths[neighbor] = depths[current] + 1
                queue.append(neighbor)
    nodes = []
    for node in G.nodes:
        depth = depths.get(node, -1)
        if node == root:
            typ = "root"
        elif str(node) in intervention_nodes:
            typ = "debunked"
        else:
            typ = "infected"
        degree = G.degree(node)
        nodes.append({
            "id": str(node),
            "depth": depth,
            "type": typ,
            "degree": degree
        })
    edges = [{"source": str(u), "target": str(v)} for u, v in G.edges]
    return {"nodes": nodes, "edges": edges}