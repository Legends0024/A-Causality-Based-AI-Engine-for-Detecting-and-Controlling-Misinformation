import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class RumourGAT(nn.Module):
    def __init__(self, in_channels=5, hidden=32, heads=4, dropout=0.3):
        super().__init__()
        self.gat1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden,
            heads=heads,
            dropout=dropout,
            concat=True
        )
        self.gat2 = GATConv(
            in_channels=hidden * heads,
            out_channels=hidden,
            heads=1,
            dropout=dropout,
            concat=False
        )
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        return self.node_classifier(x).squeeze(-1)