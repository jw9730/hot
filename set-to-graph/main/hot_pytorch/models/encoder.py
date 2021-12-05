from typing import Union

import torch.nn as nn

from .dense.attn import SelfAttn as D_SelfAttn
from .dense.kernelattn import KernelSelfAttn as D_KernelSelfAttn
from .dense.f import Apply as D_Apply, Add as D_Add
from .dense.linear import Linear as D_Linear

from .sparse.attn import SelfAttn as S_SelfAttn
from .sparse.kernelattn import KernelSelfAttn as S_KernelSelfAttn
from .sparse.f import Apply as S_Apply, Add as S_Add
from .sparse.linear import Linear as S_Linear

from ..batch.dense import Batch as D
from ..batch.sparse import Batch as S
from .common.kernel import KernelFeatureMap


class EncLayer(nn.Module):
    def __init__(self, ord_in, ord_out, dim_in, dim_qk, dim_v, dim_ff, n_heads, cfg='default', att_cfg='default',
                 dropout=0., drop_mu=0., feature_map=None, sparse=True):
        super().__init__()
        assert cfg in ('default', 'local')
        assert att_cfg in ('default', 'kernel', 'generalized_kernel')
        SelfAttn = S_SelfAttn if sparse else D_SelfAttn
        KernelSelfAttn = S_KernelSelfAttn if sparse else D_KernelSelfAttn
        Linear = S_Linear if sparse else D_Linear
        Apply = S_Apply if sparse else D_Apply
        self.add = S_Add() if sparse else D_Add()
        self.sparse = sparse

        self.ln = Apply(nn.LayerNorm(dim_in))
        if att_cfg == 'default' or ord_out == 0:
            self.attn = SelfAttn(ord_in, ord_out, dim_in, dim_qk, dim_v, n_heads, cfg, dropout, drop_mu)
        else:
            self.attn = KernelSelfAttn(ord_in, ord_out, dim_in, dim_qk, dim_v, n_heads, cfg, dropout, drop_mu, feature_map)
        self.residual = False

        self.ffn = nn.Sequential(
            Apply(nn.LayerNorm(dim_in)),
            Linear(ord_out, ord_out, dim_in, dim_ff, cfg='light'),
            Apply(nn.GELU(), skip_masking=True),
            Linear(ord_out, ord_out, dim_ff, dim_in, cfg='light'),
            Apply(nn.Dropout(dropout, inplace=True), skip_masking=True)
        )

    def forward(self, G: Union[D, S]):
        h = self.ln(G)
        h = self.attn(h)
        G = self.add(G, h) if self.residual else h
        h = self.ffn(G)
        return self.add(G, h)


class Encoder(nn.Module):
    def __init__(self, ord_in, ord_out, ord_hidden: list, dim_in, dim_out, dim_hidden, dim_qk, dim_v, dim_ff, n_heads,
                 readout_dim_qk, readout_dim_v, readout_n_heads, enc_cfg='default', att_cfg='default',
                 drop_input=0., dropout=0., drop_mu=0., sparse=True):
        super().__init__()
        Linear = S_Linear if sparse else D_Linear
        Apply = S_Apply if sparse else D_Apply
        self.sparse = sparse

        self.input = nn.Sequential(
            Linear(ord_in, ord_in, dim_in, dim_hidden, cfg='light'),
            Apply(nn.Dropout(drop_input, inplace=True), skip_masking=True)
        )

        self.feature_map = None
        self.skip_redraw_projections = True
        if att_cfg in ('kernel', 'generalized_kernel'):
            feat_dim = dim_qk // n_heads if dim_qk >= n_heads else 1
            self.feature_map = KernelFeatureMap(feat_dim, generalized_attention=(att_cfg == 'generalized_kernel'))
            self.skip_redraw_projections = False

        layers = []
        for ord1, ord2 in zip([ord_in] + ord_hidden, ord_hidden + [ord_out]):
            dim_qk_, dim_v_, n_heads_ = (dim_qk, dim_v, n_heads) if ord2 > 0 else (readout_dim_qk, readout_dim_v, readout_n_heads)
            layers.append(EncLayer(ord1, ord2, dim_hidden, dim_qk_, dim_v_, dim_ff, n_heads_, enc_cfg, att_cfg, dropout, drop_mu, self.feature_map, sparse))
        self.layers = nn.Sequential(*layers)

        self.output = Apply(
            nn.Sequential(
                nn.LayerNorm(dim_hidden),
                nn.Linear(dim_hidden, dim_out)
            )
        )

    def forward(self, G: Union[S, D]):
        assert isinstance(G, S if self.sparse else D)
        if (self.feature_map is not None) and (not self.skip_redraw_projections):
            self.feature_map.redraw_projections()
        G = self.input(G)
        G = self.layers(G)
        return self.output(G)
