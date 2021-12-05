from typing import Union, List

import torch.nn as nn

from .dense.linear import Linear as D_Linear
from .dense.f import Nonlinear as D_Nonlinear, Apply as D_Apply

from .sparse.linear import Linear as S_Linear
from .sparse.f import Nonlinear as S_Nonlinear, Apply as S_Apply

from ..batch.dense import Batch as D
from ..batch.sparse import Batch as S


class MLP(nn.Module):
    def __init__(self, ord_in, ord_out, ord_hidden: list, dim_in, dim_out, dim_hidden: Union[List, int], f='relu', dropout=0., sparse=True):
        super().__init__()
        Linear = S_Linear if sparse else D_Linear
        Nonlinear = S_Nonlinear if sparse else D_Nonlinear
        Apply = S_Apply if sparse else D_Apply
        self.sparse = sparse

        if not isinstance(dim_hidden, list):
            dim_hidden = [dim_hidden] * len(ord_hidden)
        ords = [ord_in] + ord_hidden + [ord_out]
        dims = [dim_in] + dim_hidden + [dim_out]
        od = list(zip(ords, dims))
        layers = []
        for idx, pair in enumerate(zip(od[:-1], od[1:])):
            ord1, dim1 = pair[0]
            ord2, dim2 = pair[1]
            layers.append(Linear(ord1, ord2, dim1, dim2, bias=True, cfg='default', normalize=True))
            if idx < len(od) - 2:
                layers.append(Nonlinear(f))
                layers.append(Apply(nn.Dropout(p=dropout)))
        self.layers = nn.Sequential(*layers)

    def forward(self, G: Union[D, S]):
        assert isinstance(G, S if self.sparse else D)
        return self.layers(G)
