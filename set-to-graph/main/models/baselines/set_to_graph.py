import torch
import torch.nn as nn

from .deep_sets import DeepSet
from .layers import PsiSuffix


class SetToGraph(nn.Module):
    def __init__(self, in_features, out_features, set_fn_feats, method, hidden_mlp, predict_diagonal, attention, cfg=None):
        """
        SetToGraph model.
        :param in_features: input set's number of features per data point
        :param out_features: number of output features.
        :param set_fn_feats: list of number of features for the output of each deepsets layer
        :param method: transformer method - quad, lin2 or lin5
        :param hidden_mlp: list[int], number of features in hidden layers mlp.
        :param predict_diagonal: Bool. True to predict the diagonal (diagonal needs a separate psi function).
        :param attention: Bool. Use attention in DeepSets
        :param cfg: configurations of using second bias in DeepSetLayer, normalization method and aggregation for lin5.
        """
        super(SetToGraph, self).__init__()
        assert method in ['lin2', 'lin5']

        self.method = method
        if cfg is None:
            cfg = {}
        self.agg = cfg.get('agg', torch.sum)

        self.set_model = DeepSet(in_features=in_features, feats=set_fn_feats, attention=attention, cfg=cfg)

        # Suffix - from last number of features, to 1 feature per entrance
        d2 = (2 if method == 'lin2' else 5) * set_fn_feats[-1]
        hidden_mlp = [d2] + hidden_mlp + [out_features]
        self.suffix = PsiSuffix(hidden_mlp, predict_diagonal=predict_diagonal)

    def forward(self, x):
        x = x.transpose(2, 1)  # from BxNxC to BxCxN
        u = self.set_model(x)  # Bx(out_features)xN
        n = u.shape[2]

        if self.method == 'lin2':
            m1 = u.unsqueeze(2).repeat(1, 1, n, 1)  # broadcast to rows
            m2 = u.unsqueeze(3).repeat(1, 1, 1, n)  # broadcast to cols
            block = torch.cat((m1, m2), dim=1)
        elif self.method == 'lin5':
            m1 = u.unsqueeze(2).repeat(1, 1, n, 1)  # broadcast to rows
            m2 = u.unsqueeze(3).repeat(1, 1, 1, n)  # broadcast to cols
            m3 = self.agg(u, dim=2, keepdim=True).unsqueeze(3).repeat(1, 1, n, n)  # sum over N, put on all
            m4 = u.diag_embed(dim1=2, dim2=3)  # assign values to diag only
            m5 = self.agg(u, dim=2, keepdim=True).repeat(1, 1, n).diag_embed(dim1=2, dim2=3)  # sum over N, put on diag
            block = torch.cat((m1, m2, m3, m4, m5), dim=1)
        edge_vals = self.suffix(block)  # shape (B,out_features,N,N)

        return edge_vals
