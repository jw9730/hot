import torch
import torch.nn as nn

from .layers import PsiSuffix
from .mlp import MLP


class SetToGraphSiam(nn.Module):
    def __init__(self, in_features, set_fn_feats, hidden_mlp, cfg=None):
        """
        SetToGraph model.
        :param in_features: input set's number of features per data point
        :param set_fn_feats: list of number of features for the output of each deepsets layer
        :param hidden_mlp: list[int], number of features in hidden layers mlp.
        :param cfg: configurations of mlp to end with relu and normalization method
        """
        super().__init__()

        # For comparison - in DeepSet we use 2 mlps each layer, here only 1, so double up.
        if cfg is None:
            cfg = {}
        self.set_model = MLP(in_features=in_features, feats=set_fn_feats, cfg=cfg)

        # Suffix - from last number of features, to 1 feature per entrance
        d2 = 2 * set_fn_feats[-1]
        hidden_mlp = [d2] + hidden_mlp + [1]
        self.suffix = PsiSuffix(hidden_mlp, predict_diagonal=False)

    def forward(self, x):
        x = x.transpose(2, 1)  # from B,N,C to B,C,N
        u = self.set_model(x)  # Bx(out_features)xN
        n = u.shape[2]

        m1 = u.unsqueeze(2).repeat(1, 1, n, 1)  # broadcast to rows
        m2 = u.unsqueeze(3).repeat(1, 1, 1, n)  # broadcast to cols
        block = torch.cat((m1, m2), dim=1)
        edge_vals = self.suffix(block)  # shape (B,1,N,N)

        return edge_vals
