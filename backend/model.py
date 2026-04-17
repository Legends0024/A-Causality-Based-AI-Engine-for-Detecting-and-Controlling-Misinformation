"""
model.py — RumourGAT: Graph Attention Network for misinformation detection.

Two output heads:
    predict_proba()  →  graph-level fake probability [0,1] (one scalar per cascade)
    node_scores()    →  per-node spread risk [0,1] N-vector (used by greedy optimizer)

Architecture:
    GAT Layer 1  (in_channels → hidden*heads, multi-head attention)
        → BatchNorm → ELU
    GAT Layer 2  (hidden*heads → hidden, single head)
        → BatchNorm → ELU
    Graph Embed  = concat(mean_pool, max_pool, root_node_embed)  → hidden*3
    Graph Head   = Linear(hidden*3 → 64) → BN → ReLU → Dropout
                 → Linear(64 → 32) → ReLU → Dropout
                 → Linear(32 → 1)  → sigmoid
    Node Head    = Linear(hidden → 16) → ReLU → Dropout → Linear(16 → 1) → sigmoid

Why this architecture:
    - Mean+max pooling captures both average spread behaviour and worst-case nodes
    - Root node residual gives the model direct access to the source's embedding
    - Separate node scorer allows the greedy intervention optimizer to rank nodes
      independently of the graph-level classification decision
    - BatchNorm after each GAT layer stabilises gradients on small cascade graphs
    - Dropout in both heads prevents overfitting on limited training data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class RumourGAT(nn.Module):

    def __init__(
        self,
        in_channels: int = 5,
        hidden: int = 32,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden  = hidden
        self.heads   = heads
        self.dropout = dropout

        # ── GAT encoder ───────────────────────────────────────────────────────
        self.gat1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden,
            heads=heads,
            dropout=dropout,
            concat=True,          # output: hidden * heads
        )
        self.gat2 = GATConv(
            in_channels=hidden * heads,
            out_channels=hidden,
            heads=1,
            dropout=dropout,
            concat=False,         # output: hidden
        )

        self.bn1 = nn.BatchNorm1d(hidden * heads)
        self.bn2 = nn.BatchNorm1d(hidden)

        # ── Graph-level classification head ───────────────────────────────────
        # Input: mean_pool [hidden] || max_pool [hidden] || root_embed [hidden]
        graph_embed_dim = hidden * 3

        self.graph_classifier = nn.Sequential(
            nn.Linear(graph_embed_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            # No sigmoid here — use BCEWithLogitsLoss during training
        )

        # ── Node-level spread-risk scorer ─────────────────────────────────────
        self.node_scorer = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            # No sigmoid here — applied in node_scores() at inference
        )

        # Weight initialisation
        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self):
        """Kaiming init for linear layers; constant init for BN."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ── Shared encoder ────────────────────────────────────────────────────────

    def _encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Runs two GAT layers and returns node embeddings of shape [N, hidden].

        Args:
            x          : Node feature matrix [N, in_channels]
            edge_index : Edge connectivity    [2, E]

        Returns:
            node_emb   : Node embeddings      [N, hidden]
        """
        # GAT Layer 1
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)           # [N, hidden*heads]

        # BatchNorm expects at least 2 samples during training
        if self.training and x.size(0) > 1:
            x = self.bn1(x)
        elif not self.training:
            x = self.bn1(x)

        x = F.elu(x)

        # GAT Layer 2
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)           # [N, hidden]

        if self.training and x.size(0) > 1:
            x = self.bn2(x)
        elif not self.training:
            x = self.bn2(x)

        x = F.elu(x)
        return x                               # [N, hidden]

    # ── Graph-level forward (raw logit) ───────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None,
        root_index: int = 0,
    ) -> torch.Tensor:
        """
        Returns a raw logit (before sigmoid) for the cascade.
        Use BCEWithLogitsLoss(logit, target) during training.

        Args:
            x          : Node features  [N, in_channels]
            edge_index : Edge index     [2, E]
            batch      : Batch vector   [N] — None means single graph
            root_index : Index of the root/source node in x

        Returns:
            logit      : Scalar tensor  [] or [1]
        """
        node_emb = self._encode(x, edge_index)          # [N, hidden]

        if batch is None:
            batch = torch.zeros(
                node_emb.size(0), dtype=torch.long, device=x.device
            )

        mean_emb = global_mean_pool(node_emb, batch)    # [B, hidden]
        max_emb  = global_max_pool(node_emb, batch)     # [B, hidden]

        # Root node embedding — clamp to valid range
        safe_root = min(root_index, node_emb.size(0) - 1)
        root_emb  = node_emb[safe_root].unsqueeze(0)    # [1, hidden]

        # If batched (B > 1), broadcast root_emb to match batch size
        if mean_emb.size(0) > 1:
            root_emb = mean_emb                         # fallback for batched mode

        graph_emb = torch.cat([mean_emb, max_emb, root_emb], dim=-1)  # [B, hidden*3]
        logit = self.graph_classifier(graph_emb).squeeze(-1)           # [] or [B]
        return logit

    # ── Graph-level probability (inference) ───────────────────────────────────

    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None,
        root_index: int = 0,
    ) -> float:
        """
        Returns the fake probability as a Python float in [0, 1].
        Safe to call without torch.no_grad() context (handles it internally).

        Returns:
            float  — e.g. 0.82 means 82% likely to be a rumour
        """
        self.eval()
        with torch.no_grad():
            logit = self.forward(x, edge_index, batch, root_index)
            prob  = torch.sigmoid(logit)
            # Handle both scalar and 1-element tensor
            return float(prob.squeeze().cpu().item())

    # ── Node-level spread risk (inference) ────────────────────────────────────

    def node_scores(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns per-node spread-risk probabilities in [0, 1].
        Used by the greedy causal intervention optimizer to rank nodes.

        Returns:
            probs  : [N] tensor of floats in [0, 1]
        """
        self.eval()
        with torch.no_grad():
            node_emb = self._encode(x, edge_index)                  # [N, hidden]
            logits   = self.node_scorer(node_emb).squeeze(-1)       # [N]
            return torch.sigmoid(logits)                             # [N] in [0,1]

    # ── Training loss helper ───────────────────────────────────────────────────

    def loss(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target: float,
        batch: torch.Tensor = None,
        root_index: int = 0,
        pos_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Convenience method for training.
        Uses BCEWithLogitsLoss which is numerically more stable than
        BCE(sigmoid(logit), target).

        Args:
            target     : 1.0 for rumour, 0.0 for non-rumour
            pos_weight : Weight for positive class to handle imbalanced data

        Returns:
            loss scalar tensor (call .backward() on it)

        Example:
            optimizer.zero_grad()
            loss = model.loss(x, edge_index, target=1.0)
            loss.backward()
            optimizer.step()
        """
        self.train()
        logit  = self.forward(x, edge_index, batch, root_index)
        target_tensor = torch.tensor(
            [target], dtype=torch.float, device=x.device
        )
        pw = torch.tensor([pos_weight], dtype=torch.float, device=x.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        return criterion(logit.view(1), target_tensor)

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"RumourGAT("
            f"in=5, hidden={self.hidden}, heads={self.heads}, "
            f"dropout={self.dropout} | "
            f"params={total:,} total, {trainable:,} trainable)"
        )