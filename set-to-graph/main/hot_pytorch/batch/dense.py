from typing import Union, Callable

import torch

from ..utils.dense import to_diag, get_diag, get_nondiag
from ..utils.set import get_mask, masked_fill


class Batch(object):
    A: torch.Tensor
    n_nodes: list
    device: torch.device
    mask: torch.BoolTensor
    order: int

    def __init__(self, A: torch.Tensor, n_nodes: list, mask: torch.BoolTensor = None, skip_masking: bool = False):
        """a mini-batch of dense (hyper)graphs
        :param A: Tensor([B, N^k, D])
        :param n_nodes: List([n1, ..., nb])  N >= max(sizes)
        :param mask: BoolTensor([B, N^k])
        :param skip_masking:
        """
        self.A = A
        self.n_nodes = n_nodes
        self.device = self.A.device
        self.order = len(A.size()) - 2
        self.mask = self.get_mask() if mask is None else mask
        assert self.order != 0, "zeroth-order data (N x D batched features) is expected to be handled separately"
        assert self.order in (1, 2), "we currently support dense batching up to second order"
        assert self.mask.size(1) == max(n_nodes)
        assert self.mask.size() == A.size()[:-1]
        if not skip_masking:
            # set invalid values to 0
            self.apply_mask(0)

    def __repr__(self):
        return f"Batch(A {list(self.A.size())})"

    def to(self, device: Union[str, torch.device]) -> 'Batch':
        self.A = self.A.to(device)
        self.mask = self.mask.to(device)
        self.device = self.A.device
        return self

    @torch.no_grad()
    def get_mask(self) -> torch.BoolTensor:
        node_mask = get_mask(torch.tensor(self.n_nodes, dtype=torch.long, device=self.device))  # [B, N]
        if self.order == 1:
            return node_mask  # [B, N]
        else:
            return node_mask.unsqueeze(1) & node_mask.unsqueeze(2)  # [B, N, N]

    def apply_mask(self, value=0.) -> None:
        # mask out invalid tensor elements
        self.A = masked_fill(self.A, self.mask, value)


def batch_like(G: Batch, A: torch.Tensor, skip_masking=False) -> Batch:
    # only substitute A
    return Batch(A, G.n_nodes, G.mask, skip_masking)


def t(G: Batch) -> Batch:
    # transpose
    assert G.order == 2
    return Batch(G.A.transpose(2, 1), G.n_nodes, G.mask, skip_masking=True)


def nd(G: Batch) -> Batch:
    # get non-diagonals
    assert G.order == 2
    return Batch(get_nondiag(G.A), G.n_nodes, G.mask, skip_masking=True)


def d(G: Batch, mask: torch.Tensor = None) -> Batch:
    # get diagonals
    assert G.order == 2
    if mask is not None:
        assert len(mask.size()) == 2
    return Batch(get_diag(G.A), G.n_nodes, mask, skip_masking=True)


def v2d(G: Batch, mask: torch.Tensor = None) -> Batch:
    # vectors to diagonal matrices
    assert G.order == 1
    if mask is not None:
        assert len(mask.size()) == 3
    return Batch(to_diag(G.A), G.n_nodes, mask, skip_masking=True)


def apply(G: Union[torch.Tensor, Batch], f: Callable[[torch.Tensor], torch.Tensor], skip_masking=False) -> Union[torch.Tensor, Batch]:
    if isinstance(G, torch.Tensor):
        return f(G)
    return Batch(f(G.A), G.n_nodes, G.mask, skip_masking)


def add_batch(G1: Union[torch.Tensor, Batch], G2: Union[torch.Tensor, Batch]) -> Union[torch.Tensor, Batch]:
    # add features of two batched graphs with identical edge structures
    if isinstance(G1, Batch) and isinstance(G2, Batch):
        assert G1.order == G2.order
        assert G1.n_nodes == G2.n_nodes
        assert G1.A.size() == G2.A.size()
        return Batch(G1.A + G2.A, G1.n_nodes, G1.mask, skip_masking=True)
    else:
        assert isinstance(G1, torch.Tensor) and isinstance(G2, torch.Tensor)
        assert G1.size() == G2.size()
        return G1 + G2
