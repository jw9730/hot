import torch
import torch.nn as nn

from ...batch.dense import Batch as B
from ...utils.dense import get_diag, get_nondiag


class MaxPool(nn.Module):
    def __init__(self, order):
        super().__init__()
        assert order in (1, 2), 'unsupported input order'
        self.order = order

    def forward(self, G: B) -> torch.Tensor:
        assert G.order == self.order
        # mask invalid entries with very large negative number
        G.apply_mask(-1e38)
        A = G.A
        if self.order == 1:
            assert len(A.size()) == 3  # [B, N, D]
            return A.max(1)[0]  # [B, N, D] -> [B, D]
        else:
            assert len(A.size()) == 4  # [B, N, N, D]
            n = A.size(1)
            diag = get_diag(A)  # [B, N, D]
            nondiag = get_nondiag(A)  # [B, N, N, D]
            # mask out diagonals with very large negative number
            nondiag -= torch.eye(n, device=A.device).view(1, n, n, 1) * 1e38
            diag_max = diag.max(1)[0]  # [B, D]
            nondiag_max = nondiag.flatten(1, 2).max(1)[0]  # [B, D]
            return diag_max + nondiag_max  # [B, D]


class SumPool(nn.Module):
    def __init__(self, order):
        super().__init__()
        assert order > 0
        self.order = order

    def forward(self, G: B) -> torch.Tensor:
        assert G.order == self.order
        # mask invalid entries with zero
        G.apply_mask(0)
        A = G.A  # [B, N^k, D]
        bsize, n, d = A.size(0), A.size(1), A.size(-1)
        A = A.view(bsize, n ** self.order, d)  # [B, (N^k), D]
        return A.sum(1)  # [B, D]


class AvgPool(nn.Module):
    def __init__(self, order):
        super().__init__()
        assert order in (1, 2), 'unsupported input order'
        self.order = order
        self.eps = 1e-5

    def forward(self, G: B) -> torch.Tensor:
        assert G.order == self.order
        # mask invalid entries with zero
        G.apply_mask(0)
        A = G.A  # [B, N^k, D]
        # will normalize with instance sizes
        n_vec = torch.tensor(G.n_nodes, device=A.device).float().unsqueeze(-1)  # [B, 1]
        if self.order == 1:
            assert len(A.size()) == 3  # [B, N, D]
            feat = A.sum(1)  # [B, D]
            n_inv = (n_vec + self.eps).pow(-1)  # [B, 1]
            feat_avg = feat * n_inv  # [B, D]
            return feat_avg
        else:
            assert len(A.size()) == 4  # [B, N, N, D]
            diag_sum = get_diag(A).sum(1)  # [B, N, D] -> [B, D]
            nondiag_sum = get_nondiag(A).flatten(1, 2).sum(1)  # [B, N, N, D] -> [B, D]
            n_inv = (n_vec + self.eps).pow(-1)  # [B, 1]
            diag_avg = diag_sum * n_inv  # [B, D]
            e_inv = (n_vec.pow(2) - n_vec + self.eps).pow(-1)  # [B, 1]
            nondiag_avg = nondiag_sum * e_inv  # [B, D]
            return diag_avg + nondiag_avg
