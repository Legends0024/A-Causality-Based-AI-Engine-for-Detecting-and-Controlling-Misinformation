import networkx as nx


def _resolve_graph_node(G, node):
    if node in G.nodes:
        return node
    node_str = str(node)
    if node_str in G.nodes:
        return node_str
    try:
        node_int = int(node)
    except (TypeError, ValueError):
        node_int = None
    if node_int in G.nodes:
        return node_int
    return node


def extract_node_features(G, node, root, max_depth: int = 10) -> list:
    root_node = _resolve_graph_node(G, root)
    total_nodes = max(len(G.nodes), 1)
    is_directed = G.is_directed()
    try:
        path  = nx.shortest_path(G, source=root_node, target=node)
        depth = len(path) - 1
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        depth = -1
    depth_norm   = depth / max_depth if depth >= 0 else 0.0
    degree       = G.degree(node)
    in_deg       = G.in_degree(node)  if is_directed else 0
    out_deg      = G.out_degree(node) if is_directed else 0
    degree_norm  = degree  / total_nodes
    in_deg_norm  = in_deg  / total_nodes
    out_deg_norm = out_deg / total_nodes
    is_root      = 1.0 if str(node) == str(root_node) else 0.0
    return [depth_norm, degree_norm, in_deg_norm, out_deg_norm, is_root]


def bfs_depths(G, root) -> dict:
    root_node = _resolve_graph_node(G, root)
    if root_node not in G.nodes:
        return {}
    depths  = {root_node: 0}
    visited = {root_node}
    queue   = [root_node]
    successor_fn = G.successors if G.is_directed() else G.neighbors
    while queue:
        current = queue.pop(0)
        for neighbour in successor_fn(current):
            if neighbour not in visited:
                visited.add(neighbour)
                depths[neighbour] = depths[current] + 1
                queue.append(neighbour)
    return depths


def graph_to_json(G, root, intervention_nodes=None) -> dict:
    root_node = _resolve_graph_node(G, root)
    intervention_set = {str(n) for n in intervention_nodes} if intervention_nodes else set()
    depths = bfs_depths(G, root_node)
    nodes = []
    for node in G.nodes:
        depth    = depths.get(node, -1)
        node_str = str(node)
        if str(node) == str(root_node):
            node_type = "root"
        elif node_str in intervention_set:
            node_type = "debunked"
        else:
            node_type = "infected"
        nodes.append({"id": node_str, "depth": depth, "type": node_type, "degree": G.degree(node)})
    edges = [{"source": str(u), "target": str(v)} for u, v in G.edges]
    return {"nodes": nodes, "edges": edges}
