#!/usr/bin/env python3
"""
Joint GNN predictor for link prediction + attribute prediction.

Wraps ANY existing GNN model (GATv2, GCN, NCN, NCNC, NeoGNN, BUDDY) and
adds an attribute prediction head on top of the same encoder.

  - Link decoder: the original model's decode() (specialized per architecture)
  - Attr decoder: a simple MLP EdgePredictor on shared embeddings

Training: link_loss (BCE, pos+neg) + attr_weight * attr_loss (MSE, pos only)
Inference: σ(link) × σ(attr)

Heckman selection correction (optional):
  When use_heckman=True, the attr head receives σ(link_logits).detach() as an
  extra input feature (control function), providing explicit bias correction
  for sample selection bias in reported evaluations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn_link_predictor import EdgePredictor


class GNNJointPredictor(nn.Module):
    """Generic joint predictor wrapping any GNN link predictor.

    Args:
        base_model: Any model with encode(x, edge_index) -> z and decode(z, edge_index) -> logits.
                    Its decode() is used as the link head.
        out_channels: Dimension of z (encoder output). If None, auto-detected.
        attr_decoder: Type of attr decoder ("mlp" or "bilinear").
        dropout: Dropout for attr head.
        use_heckman: If True, feed σ(link_logits) as control function to attr head.
    """

    def __init__(self, base_model: nn.Module, out_channels: int = None, attr_decoder: str = "mlp",
                 dropout: float = 0.2, use_heckman: bool = False):
        super().__init__()
        self.base_model = base_model
        self.use_heckman = use_heckman

        # Auto-detect output dimension
        if out_channels is None:
            if hasattr(base_model, "encoder") and hasattr(base_model.encoder, "out_channels"):
                out_channels = base_model.encoder.out_channels
            else:
                raise ValueError("Cannot auto-detect out_channels; pass it explicitly.")

        extra_dim = 1 if use_heckman else 0
        self.attr_head = EdgePredictor(out_channels, mode=attr_decoder, dropout=dropout, extra_dim=extra_dim)

    def encode(self, x, edge_index, **kwargs):
        return self.base_model.encode(x, edge_index, **kwargs)

    def decode_link(self, z, edge_index):
        """Link prediction logits (uses base model's specialized decoder)."""
        return self.base_model.decode(z, edge_index)

    def decode_attr(self, z, edge_index, link_logits=None):
        """Attribute prediction logits.

        Args:
            z: Node embeddings.
            edge_index: Edge indices to predict on.
            link_logits: Pre-computed link logits (optional). If use_heckman=True
                         and link_logits is None, they are computed automatically.
        """
        if self.use_heckman:
            if link_logits is None:
                link_logits = self.decode_link(z, edge_index)
            # Control function: σ(link_logits), detached to stop gradient
            p = torch.sigmoid(link_logits.detach()).unsqueeze(-1)
            return self.attr_head(z, edge_index, extra_features=p)
        return self.attr_head(z, edge_index)

    def decode(self, z, edge_index):
        """Combined score: σ(link) × σ(attr_corrected)."""
        link_logits = self.decode_link(z, edge_index)
        link_score = torch.sigmoid(link_logits)
        attr_logits = self.decode_attr(z, edge_index, link_logits=link_logits)
        attr_score = torch.sigmoid(torch.clamp(attr_logits, -10, 10))
        return link_score * attr_score
