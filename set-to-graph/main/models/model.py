import torch
import torch.nn as nn

from hot_pytorch.models import Encoder
from hot_pytorch.batch.dense import Batch
from .baselines.layers import PsiSuffix


class EncoderS2G(nn.Module):
    def __init__(self, dim_in, dim_out, set_fn_feats, dim_qk, dim_v, dim_ff, n_heads, use_kernel, drop_input, dropout,
                 hidden_mlp, predict_diagonal):
        super().__init__()
        # layer 1 + 2: set-to-set + set-to-graph
        dim_hidden = set_fn_feats[0]
        for h in set_fn_feats:
            assert h == dim_hidden, 'transformer only allows constant hidden dimensions'
        ord_hidden = [1] * len(set_fn_feats)
        self.enc = Encoder(1, 2, ord_hidden, dim_in, dim_hidden, dim_hidden, dim_qk, dim_v, dim_ff, n_heads, 0, 0, 0,
                           'default', 'generalized_kernel' if use_kernel else 'default', drop_input, dropout, sparse=False)
        # layer 3: graph-to-graph
        hidden_mlp = [dim_hidden] + hidden_mlp + [dim_out]
        self.suffix = PsiSuffix(hidden_mlp, predict_diagonal=predict_diagonal)

    def forward(self, x: torch.Tensor):
        # x: [B, N, C]
        G = Batch(x, n_nodes=[x.size(1)] * len(x))
        G = self.enc(G)  # [B, N, N, D]
        x = G.A.permute(0, 3, 1, 2)  # [B, D, N, N]
        edge_vals = self.suffix(x)
        return edge_vals
