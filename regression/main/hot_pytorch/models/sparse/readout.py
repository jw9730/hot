import torch
import torch.nn as nn

from ...batch.sparse import Batch as B
from ...utils.sparse import get_diag, get_nondiag


class MaxPool(nn.Module):
    def __init__(self, order):
        super().__init__()
        assert order in (1, 2), 'unsupported input order'
        self.order = order

    def forward(self, G: B):
        assert G.order == self.order
        # mask invalid entries with very large negative number
        G.apply_mask(-1e38)
        values = G.values
        if self.order == 1:
            return values.max(1)[0]  # [B, N, D] -> [B, D]
        else:
            diag = get_diag(G.indices, G.values, G.n_nodes, G.mask, G.node_mask, G.node_ofs)
            diag = diag.masked_fill_(~G.node_mask.unsqueeze(-1), -1e38)  # [B, N, D]
            # mask out diagonals with very large negative number
            nondiag = get_nondiag(G.indices, G.values, G.mask)  # [B, |E|, D]
            diag_mask = G.indices[..., 0] == G.indices[..., 1]  # [B, |E|]
            nondiag = nondiag.masked_fill_(~G.mask.unsqueeze(-1), -1e38)
            nondiag = nondiag.masked_fill_(diag_mask.unsqueeze(-1), -1e38)
            diag_max = diag.max(1)[0]  # [B, D]
            nondiag_max = nondiag.max(1)[0]  # [B, D]
            return diag_max + nondiag_max  # [B, D]


class SumPool(nn.Module):
    def __init__(self, order):
        super().__init__()
        assert order in (1, 2), 'unsupported input order'
        self.order = order

    def forward(self, G: B):
        assert G.order == self.order
        # mask invalid entries with zero
        G.apply_mask(0)
        return G.values.sum(1)  # [B, D]


class AvgPool(nn.Module):
    def __init__(self, order):
        super().__init__()
        assert order in (1, 2), 'unsupported input order'
        self.order = order

    def forward(self, G: B):
        assert G.order == self.order
        # mask invalid entries with zero
        G.apply_mask(0)
        values = G.values
        if self.order == 1:
            n_nodes = torch.tensor(G.n_nodes, device=values.device)  # [B,]
            n_inv = (n_nodes + 1e-5).pow(-1).unsqueeze(1)  # [B, 1]
            return n_inv * values.sum(1)  # [B, D]
        else:
            n_nodes = torch.tensor(G.n_nodes, device=values.device)  # [B,]
            n_edges = torch.tensor(G.n_edges, device=values.device)  # [B,]
            n_inv = (n_nodes + 1e-5).pow(-1).unsqueeze(1)  # [B, 1]
            e_inv = (n_edges - n_nodes + 1e-5).pow(-1).unsqueeze(1)  # [B, 1]
            diag = get_diag(G.indices, G.values, G.n_nodes, G.mask, G.node_mask, G.node_ofs)
            diag = diag.masked_fill_(~G.node_mask.unsqueeze(-1), 0)  # [B, N, D]
            nondiag = get_nondiag(G.indices, G.values, G.mask)  # [B, |E|, D]
            diag_mask = G.indices[..., 0] == G.indices[..., 1]  # [B, |E|]
            nondiag = nondiag.masked_fill_(~G.mask.unsqueeze(-1), 0)
            nondiag = nondiag.masked_fill_(diag_mask.unsqueeze(-1), 0)
            diag_avg = diag.sum(1) * n_inv  # [B, D]
            nondiag_avg = nondiag.sum(1) * e_inv  # [B, D]
            return diag_avg + nondiag_avg  # [B, D]
