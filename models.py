"""
models/
───────
Three-component model hierarchy:

    BaseTemporalEncoder          (ABC — swap-ready interface)
    ├── LSTMTemporalEncoder      (default)
    └── GRUTemporalEncoder       (drop-in replacement)

    SpatialGNN                   (GINEConv + LayerNorm stack)

    SpatioTemporalLeakDetector   (Super-Module — joint training)

The Super-Module's forward() accepts a PyG Batch object directly, so the
training loop stays clean:

    node_logits, graph_logit = model(batch)

Backpropagation flows:  GNN loss → GNN → LSTM embeddings → LSTM weights.
This lets the temporal encoder learn representations that are *spatially aware*.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv, global_mean_pool


# ══════════════════════════════════════════════════════════════════════════════
# 1 ── Temporal Encoders
# ══════════════════════════════════════════════════════════════════════════════

class BaseTemporalEncoder(nn.Module, ABC):
    """
    Swap-ready interface for temporal encoders.

    Contract:
        input  : x [total_nodes, T, input_dim]
                 (after PyG batching, nodes from all graphs in the batch
                  are stacked on the first axis)
        output : embeddings [total_nodes, embedding_dim]

    To swap in a Transformer or GRU, subclass this and override `forward`.
    Then pass your class to SpatioTemporalLeakDetector(cfg, encoder=YourClass(...)).
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


class LSTMTemporalEncoder(BaseTemporalEncoder):
    """
    2-layer bidirectional-optional LSTM that compresses a rolling window of
    node time-series into a 32-dim latent vector per node.

    Architecture:
        LSTM(input_dim → hidden_dim, 2 layers)
          → take h_n from final layer
          → Linear(hidden_dim → embedding_dim)
          → LayerNorm

    Args:
        input_dim     : features per timestep (1=pressure, 2=pressure+demand)
        hidden_dim    : LSTM hidden state width
        embedding_dim : output width (= GCN input width)
        num_layers    : stacked LSTM layers
        dropout       : applied between LSTM layers (ignored when num_layers=1)
    """

    def __init__(
        self,
        input_dim:     int   = 1,
        hidden_dim:    int   = 64,
        embedding_dim: int   = 32,
        num_layers:    int   = 2,
        dropout:       float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [total_nodes, T, input_dim]
        → [total_nodes, embedding_dim]
        """
        _, (h_n, _) = self.lstm(x)      # h_n: [num_layers, total_nodes, hidden]
        h_last = h_n[-1]                 # [total_nodes, hidden]
        return self.projector(h_last)    # [total_nodes, embedding_dim]


class GRUTemporalEncoder(BaseTemporalEncoder):
    """
    Drop-in GRU replacement.  Identical interface to LSTMTemporalEncoder.
    Use for ablation studies or when memory is tight.
    """

    def __init__(
        self,
        input_dim:     int   = 1,
        hidden_dim:    int   = 64,
        embedding_dim: int   = 32,
        num_layers:    int   = 2,
        dropout:       float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        return self.projector(h_n[-1])


# ══════════════════════════════════════════════════════════════════════════════
# 2 ── Spatial GNN
# ══════════════════════════════════════════════════════════════════════════════

class SpatialGNN(nn.Module):
    """
    Multi-layer Graph Isomorphism Network with Edge features (GINEConv).
    Operates on LSTM embeddings and produces both node-level and graph-level
    predictions.

    Why GINEConv?
      - More expressive than GCNConv (provably as powerful as 1-WL test).
      - Accepts edge_attr (mean flow per pipe) natively.
      - Allows us to leverage hydraulic flow information during message passing.

    Args:
        embedding_dim  : input feature dim  (= LSTM embedding_dim)
        hidden_dim     : hidden channel width inside each conv layer
        edge_feat_dim  : dimension of edge_attr (mean flow → 1)
        num_layers     : number of GINEConv layers
        dropout        : feature dropout between layers
    """

    def __init__(
        self,
        embedding_dim: int   = 32,
        hidden_dim:    int   = 64,
        edge_feat_dim: int   = 1,
        num_layers:    int   = 2,
        dropout:       float = 0.2,
    ):
        super().__init__()
        self.dropout = dropout
        self.convs   = nn.ModuleList()
        self.norms   = nn.ModuleList()

        in_dim = embedding_dim
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=edge_feat_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim

        # Shared classification head used for both node-level and graph-level
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),   # raw logit
        )

    def forward(
        self,
        x:          torch.Tensor,  # [total_nodes, embedding_dim]
        edge_index: torch.Tensor,  # [2, total_edges]
        edge_attr:  torch.Tensor,  # [total_edges, edge_feat_dim]
        batch_vec:  torch.Tensor,  # [total_nodes] PyG batch membership
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            node_logits  : [total_nodes, 1]  raw per-node scores
            graph_logits : [B]               raw per-graph score (mean-pooled)
        """
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index, edge_attr)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        node_logits = self.node_head(h)                      # [total_nodes, 1]

        graph_h     = global_mean_pool(h, batch_vec)         # [B, hidden_dim]
        graph_logits = self.node_head(graph_h).squeeze(-1)   # [B]

        return node_logits, graph_logits


# ══════════════════════════════════════════════════════════════════════════════
# 3 ── Super-Module
# ══════════════════════════════════════════════════════════════════════════════

class SpatioTemporalLeakDetector(nn.Module):
    """
    End-to-end jointly-trained spatio-temporal leak detector.

    Data flow:
        batch.x  [total_nodes, T, F]
          │
          ▼  LSTMTemporalEncoder
        embeddings  [total_nodes, embedding_dim]
          │
          ▼  SpatialGNN (GINEConv × num_layers)
        node_logits  [total_nodes, 1]
        graph_logits [B]

    Gradient path: GNN loss → GNN → embeddings → LSTM
    This ensures the LSTM learns to produce embeddings that are *useful
    for spatial leak localisation*, not just temporal compression.

    Swapping the encoder:
        from models import GRUTemporalEncoder, SpatioTemporalLeakDetector
        model = SpatioTemporalLeakDetector(
            cfg,
            encoder=GRUTemporalEncoder(input_dim=cfg.input_dim, ...)
        )
    """

    def __init__(
        self,
        cfg,                                   # PipelineConfig
        encoder: Optional[BaseTemporalEncoder] = None,
    ):
        super().__init__()

        self.temporal_encoder: BaseTemporalEncoder = encoder or LSTMTemporalEncoder(
            input_dim     = cfg.input_dim,
            hidden_dim    = cfg.lstm_hidden,
            embedding_dim = cfg.embedding_dim,
            num_layers    = cfg.lstm_layers,
            dropout       = cfg.lstm_dropout,
        )

        self.spatial_gnn = SpatialGNN(
            embedding_dim = cfg.embedding_dim,
            hidden_dim    = cfg.gnn_hidden,
            edge_feat_dim = cfg.edge_feat_dim,
            num_layers    = cfg.gnn_layers,
            dropout       = cfg.gnn_dropout,
        )

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch : PyG Batch (collation of Data objects from LeakWindowDataset)
                    batch.x          [total_nodes, T, input_dim]
                    batch.edge_index [2, total_edges]
                    batch.edge_attr  [total_edges, edge_feat_dim]
                    batch.batch      [total_nodes]  graph membership

        Returns:
            node_logits  : [total_nodes, 1]  raw per-node leak scores
            graph_logits : [B]               raw per-graph leak score
        """
        # ── Stage 1: Temporal encoding ────────────────────────────────────
        #   batch.x is already [total_nodes, T, F] after PyG batching
        node_embeddings = self.temporal_encoder(batch.x)   # [total_nodes, emb_dim]

        # ── Stage 2: Spatial reasoning ────────────────────────────────────
        node_logits, graph_logits = self.spatial_gnn(
            x          = node_embeddings,
            edge_index = batch.edge_index,
            edge_attr  = batch.edge_attr,
            batch_vec  = batch.batch,
        )

        return node_logits, graph_logits

    # ── Convenience ─────────────────────────────────────────────────────────

    def encode_nodes(self, batch: Batch) -> torch.Tensor:
        """
        Extract raw LSTM embeddings without running the GNN.
        Useful for embedding export / downstream tasks.

        Returns: [total_nodes, embedding_dim]
        """
        with torch.no_grad():
            return self.temporal_encoder(batch.x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
