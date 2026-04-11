import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class RumourGAT(nn.Module):
    """
    Graph Attention Network for rumour/misinformation detection.

    Architecture:
        GAT Layer 1  →  multi-head attention (captures local neighbourhood)
        GAT Layer 2  →  single-head refinement
        Graph Pooling →  mean + max pooled graph-level embedding
                         + root node embedding concatenated
        Classifier   →  MLP → single logit (fake probability)

    Two output heads:
        forward() / predict_proba()  →  graph-level fake probability (one per cascade)
        node_scores()                →  per-node spread risk (used by greedy optimizer)
    """

    def __init__(self, in_channels: int = 5, hidden: int = 32, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout

        # ── GAT Layers ──────────────────────────────────────────────────────
        self.gat1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden,
            heads=heads,
            dropout=dropout,
            concat=True,         # output: hidden * heads
        )
        self.gat2 = GATConv(
            in_channels=hidden * heads,
            out_channels=hidden,
            heads=1,
            dropout=dropout,
            concat=False,        # output: hidden
        )

        # ── Batch Normalisation ──────────────────────────────────────────────
        self.bn1 = nn.BatchNorm1d(hidden * heads)
        self.bn2 = nn.BatchNorm1d(hidden)

        # ── Graph-level Classifier ───────────────────────────────────────────
        # mean_pool(hidden) + max_pool(hidden) + root_embed(hidden) = hidden * 3
        graph_embed_dim = hidden * 3
        self.graph_classifier = nn.Sequential(
            nn.Linear(graph_embed_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),   # single logit → sigmoid → fake probability
        )

        # ── Node-level Scorer (used by greedy intervention optimizer) ────────
        self.node_scorer = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def _encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Shared encoder — returns per-node embeddings of shape [N, hidden]."""
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)

        return x  # [N, hidden]

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None,
        root_index: int = 0,
    ) -> torch.Tensor:
        """
        Graph-level forward pass.

        Args:
            x           : Node feature matrix [N, in_channels]
            edge_index  : Edge connectivity    [2, E]
            batch       : Batch vector [N] mapping each node to its graph.
                          If None (single graph), all nodes belong to graph 0.
            root_index  : Index of the root / source node (default 0).

        Returns:
            logit : Raw fake-news logit per graph — shape [1] for single graph.
                    Apply sigmoid to get probability.
        """
        node_emb = self._encode(x, edge_index)  # [N, hidden]

        if batch is None:
            batch = torch.zeros(node_emb.size(0), dtype=torch.long, device=x.device)

        mean_emb = global_mean_pool(node_emb, batch)   # [B, hidden]
        max_emb  = global_max_pool(node_emb, batch)    # [B, hidden]

        # Root node carries the most informative signal in a rumour cascade
        root_emb = node_emb[root_index].unsqueeze(0)   # [1, hidden]
        if mean_emb.size(0) > 1:
            root_emb = mean_emb  # batched mode fallback

        graph_emb = torch.cat([mean_emb, max_emb, root_emb], dim=-1)  # [B, hidden*3]
        logit = self.graph_classifier(graph_emb).squeeze(-1)           # [B] or scalar
        return logit

    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None,
        root_index: int = 0,
    ) -> torch.Tensor:
        """
        Returns graph-level fake probability in [0, 1].

        Usage:
            prob = model.predict_proba(x, edge_index)
            is_fake = prob.item() > 0.65
        """
        with torch.no_grad():
            logit = self.forward(x, edge_index, batch, root_index)
            return torch.sigmoid(logit)

    def node_scores(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns per-node spread-risk scores in [0, 1].

        Used by the greedy causal intervention optimizer in inference.py
        to decide which nodes to suppress.

        Args:
            x          : Node feature matrix [N, in_channels]
            edge_index : Edge connectivity   [2, E]

        Returns:
            scores : Per-node risk scores [N] in [0, 1]
        """
        with torch.no_grad():
            node_emb = self._encode(x, edge_index)           # [N, hidden]
            scores   = self.node_scorer(node_emb).squeeze(-1) # [N]
            return torch.sigmoid(scores)