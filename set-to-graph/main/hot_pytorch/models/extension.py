"""An extension specifically designed for k-uniform hyperedge prediction"""
from typing import Union, List

import torch
import torch.nn as nn

from .uniform.kernelattn import KernelSelfAttn
from .uniform.f import Nonlinear, Add
from .uniform.linear import Linear
from .common.kernel import KernelFeatureMap
from .sparse.masksum import count_unique


def _check_indices(indices: torch.LongTensor):
    if indices is not None:
        k = indices.size(1)
        assert (count_unique(indices) == k).all(), 'This extension is only for 1->k-uniform, but given indices contain loops'


class EncLayer(nn.Module):
    def __init__(self, ord_in, ord_out, dim_in, dim_qk, dim_v, dim_ff, n_heads, cfg='default', att_cfg='default',
                 dropout=0., drop_mu=0., feature_map=None, sparse=False):
        super().__init__()
        assert cfg == 'default' and att_cfg == 'generalized_kernel'
        assert ord_in == 1
        self.ord_out = ord_out
        self.add = Add()
        self.ln = nn.LayerNorm(dim_in)
        self.attn = KernelSelfAttn(1, ord_out, dim_in, dim_qk, dim_v, n_heads, cfg, dropout, drop_mu, feature_map)
        self.residual = False
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, dim_in),
            nn.Dropout(dropout, inplace=True)
        )

    def forward(self, x: torch.Tensor, indices: torch.LongTensor = None):
        h = self.ln(x)
        h = self.attn(h, indices)
        x = self.add(x, h) if self.residual else h
        h = self.ffn(x)
        return self.add(x, h)


class Encoder(nn.Module):
    def __init__(self, ord_in, ord_out, ord_hidden: list, dim_in, dim_out, dim_hidden, dim_qk, dim_v, dim_ff, n_heads,
                 readout_dim_qk, readout_dim_v, readout_n_heads, enc_cfg='default', att_cfg='generalized_kernel',
                 drop_input=0., dropout=0., drop_mu=0., sparse=False):
        super().__init__()
        assert enc_cfg == 'default' and att_cfg == 'generalized_kernel'
        assert ord_out > 1

        self.input = nn.Sequential(
            Linear(1, 1, dim_in, dim_hidden, cfg='light'),
            nn.Dropout(drop_input, inplace=True)
        )

        feat_dim = dim_qk // n_heads if dim_qk >= n_heads else 1
        self.feature_map = KernelFeatureMap(feat_dim, generalized_attention=(att_cfg == 'generalized_kernel'))
        self.skip_redraw_projections = False

        layers = []
        for ord1, ord2 in zip([ord_in] + ord_hidden, ord_hidden + [ord_out]):
            assert ord1 == 1
            layers.append(EncLayer(1, ord2, dim_hidden, dim_qk, dim_v, dim_ff, n_heads, enc_cfg, att_cfg, dropout, drop_mu, self.feature_map, sparse))
        self.layers = nn.ModuleList(layers)

        self.output = nn.Sequential(
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_out)
        )

    def forward(self, x: torch.Tensor, indices: torch.LongTensor):
        _check_indices(indices)
        if (self.feature_map is not None) and (not self.skip_redraw_projections):
            self.feature_map.redraw_projections()
        x = self.input(x)
        for layer in self.layers:
            x = layer(x) if layer.ord_out == 1 else layer(x, indices)
        return self.output(x)


class MLP(nn.Module):
    def __init__(self, ord_in, ord_out, ord_hidden: list, dim_in, dim_out, dim_hidden: Union[List, int], f='relu', dropout=0., sparse=False):
        super().__init__()
        assert ord_out > 1
        if not isinstance(dim_hidden, list):
            dim_hidden = [dim_hidden] * len(ord_hidden)
        ords = [ord_in] + ord_hidden + [ord_out]
        dims = [dim_in] + dim_hidden + [dim_out]
        od = list(zip(ords, dims))
        layers = []
        for idx, pair in enumerate(zip(od[:-1], od[1:])):
            ord1, dim1 = pair[0]
            ord2, dim2 = pair[1]
            assert ord1 == 1
            layer = nn.ModuleList()
            layer.append(Linear(1, ord2, dim1, dim2, bias=True, cfg='default', normalize=True))
            if idx < len(od) - 2:
                layer.append(Nonlinear(f))
                layer.append(nn.Dropout(p=dropout))
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, indices: torch.LongTensor):
        _check_indices(indices)
        for layer in self.layers:
            if len(layer) == 1:
                x = layer[0](x) if layer[0].weight.ord_out == 1 else layer[0](x, indices)
            else:
                x = layer[0](x) if layer[0].weight.ord_out == 1 else layer[0](x, indices)
                x = layer[1](x)
                x = layer[2](x)
        return x
