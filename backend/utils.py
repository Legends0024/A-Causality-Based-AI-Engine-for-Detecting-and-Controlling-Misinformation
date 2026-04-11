import networkx as nx


def extract_node_features(G, node, root, max_depth: int = 10) -> list:
    """
    Returns a 5-dimensional feature vector for a node in the cascade graph.

    Features:
        [0] depth_norm    — BFS depth from root, normalised by max_depth
        [1] degree_norm   — total degree / total_nodes
        [2] in_deg_norm   — in-degree  / total_nodes (0 for undirected)
        [3] out_deg_norm  — out-degree / total_nodes (0 for undirected)
        [4] is_root       — 1.0 if this node is the root, else 0.0
    """
    total_nodes = max(len(G.nodes), 1)
    is_directed = G.is_directed()

    # Depth via shortest path from root
    try:
        path  = nx.shortest_path(G, source=root, target=node)
        depth = len(path) - 1
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        depth = -1

    depth_norm = depth / max_depth if depth >= 0 else 0.0

    degree = G.degree(node)

    if is_directed:
        in_deg  = G.in_degree(node)
        out_deg = G.out_degree(node)
    else:
        in_deg  = 0
        out_deg = 0

    degree_norm  = degree  / total_nodes
    in_deg_norm  = in_deg  / total_nodes
    out_deg_norm = out_deg / total_nodes
    is_root      = 1.0 if node == root else 0.0

    return [depth_norm, degree_norm, in_deg_norm, out_deg_norm, is_root]


def bfs_depths(G, root) -> dict:
    """
    BFS from root through G.successors().

    Returns:
        dict {node -> depth}  (root=0, unreachable nodes absent)
    """
    depths  = {root: 0}
    visited = {root}
    queue   = [root]

    while queue:
        current = queue.pop(0)
        for neighbour in G.successors(current):
            if neighbour not in visited:
                visited.add(neighbour)
                depths[neighbour] = depths[current] + 1
                queue.append(neighbour)

    return depths


def graph_to_json(G, root, intervention_nodes=None) -> dict:
    """
    Converts a NetworkX DiGraph to a JSON-serialisable dict
    for the frontend D3 / SVG renderer.

    Args:
        G                  : NetworkX graph
        root               : Root node ID
        intervention_nodes : Set/list of suppressed node IDs (int or str)

    Returns:
        {
          "nodes": [{"id", "depth", "type", "degree"}, ...],
          "edges": [{"source", "target"}, ...]
        }

    Node types:
        "root"     — the source node
        "debunked" — suppressed by the intervention optimizer
        "infected" — all other nodes
    """
    # Normalise to strings so int/str IDs both match
    if intervention_nodes is None:
        intervention_set = set()
    else:
        intervention_set = {str(n) for n in intervention_nodes}

    depths = bfs_depths(G, root)

    nodes = []
    for node in G.nodes:
        depth    = depths.get(node, -1)
        node_str = str(node)

        if node == root:
            node_type = "root"
        elif node_str in intervention_set:
            node_type = "debunked"
        else:
            node_type = "infected"

        nodes.append({
            "id"    : node_str,
            "depth" : depth,
            "type"  : node_type,
            "degree": G.degree(node),
        })

    edges = [
        {"source": str(u), "target": str(v)}
        for u, v in G.edges
    ]

    return {"nodes": nodes, "edges": edges}