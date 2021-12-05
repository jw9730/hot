from typing import Union, Callable

import torch
import numpy as np

from ..utils.sparse import to_diag, get_diag, get_nondiag, get_transpose_info, do_transpose
from ..utils.set import get_mask, masked_fill


class Batch(object):
    indices: Union[None, torch.LongTensor]
    values: torch.Tensor
    n_nodes: list
    n_edges: Union[None, list]
    device: torch.device
    mask: torch.BoolTensor
    node_mask: torch.BoolTensor
    order: int
    t_indices: Union[None, torch.LongTensor]
    t_mask: Union[None, torch.BoolTensor]
    node_ofs: torch.LongTensor

    def __init__(self, indices: Union[None, torch.LongTensor], values: torch.Tensor,
                 n_nodes: list, n_edges: Union[None, list], mask: torch.BoolTensor = None, skip_masking: bool = False,
                 t_indices=None, t_mask=None, node_mask: torch.BoolTensor = None, node_ofs: torch.LongTensor = None):
        """a mini-batch of sparse (hyper)graphs
        :param indices: LongTensor([B, |E|, k])
        :param values: Tensor([B, |E|, D])
        :param n_nodes: List([n1, ..., nb])
        :param n_edges: List([|E1|, ..., |Eb|])
        :param mask: BoolTensor([B, |E|])
        :param skip_masking:
        :param t_indices: LongTensor([B, |E|])
        :param t_mask: BoolTensor([B, |E|])
        :param node_mask: BoolTensor([B, N])
        :param node_ofs: LongTensor([B,])
        """
        # caution: to reduce overhead, we assume a specific organization of indices: see comment in get_diag()
        # we also assume that indices are already well-masked (invalid entries are zero): see comment in self.apply_mask()
        self.indices = indices  # [B, |E|, k] or None
        self.values = values  # [B, |E|, D]
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.device = values.device
        self.order = 1 if indices is None else indices.size(-1)
        assert self.order in (1, 2)
        self.node_mask = get_mask(torch.tensor(n_nodes, dtype=torch.long, device=self.device)) if node_mask is None else node_mask  # [B, N]
        if self.order == 1:
            self.mask = self.node_mask
        else:
            self.mask = get_mask(torch.tensor(n_edges, dtype=torch.long, device=self.device)) if mask is None else mask  # [B, |E|]
        if not skip_masking:
            # set invalid values to 0
            self.apply_mask(0)
        if self.order == 2 and t_indices is None and t_mask is None:
            # precompute indices for transpose
            self.t_indices, self.t_mask = get_transpose_info(indices, self.mask)
        else:
            self.t_indices, self.t_mask = t_indices, t_mask
        self.node_ofs = torch.tensor(np.cumsum([0] + self.n_nodes[:-1]), dtype=torch.long, device=self.device) if node_ofs is None else node_ofs  # [B,]

    def __repr__(self):
        return f"Batch(indices {list(self.indices.size())}, values {list(self.values.size())}"

    def to(self, device: Union[str, torch.device]) -> 'Batch':
        if self.indices is not None:
            self.indices = self.indices.to(device)
            self.t_indices = self.t_indices.to(device)
            self.t_mask = self.t_mask.to(device)
        self.values = self.values.to(device)
        self.mask = self.mask.to(device)
        self.node_mask = self.node_mask.to(device)
        self.node_ofs = self.node_ofs.to(device)
        self.device = self.values.device
        return self

    def apply_mask(self, value=0.) -> None:
        # mask out invalid tensor elements
        # caution: to avoid overhead, we assume that indices are already well-masked (invalid entries are zero)
        self.values = masked_fill(self.values, self.mask, value)


def batch_like(G: Batch, values: torch.Tensor, skip_masking=False) -> Batch:
    # only substitute values
    return Batch(G.indices, values, G.n_nodes, G.n_edges, G.mask, skip_masking, G.t_indices, G.t_mask, G.node_mask, G.node_ofs)


def t(G: Batch) -> Batch:
    # transpose
    assert G.order == 2
    v_t = do_transpose(G.values, G.t_indices, G.t_mask)
    return batch_like(G, v_t, skip_masking=True)


def nd(G: Batch) -> Batch:
    # get non-diagonals
    assert G.order == 2
    v_nd = get_nondiag(G.indices, G.values, G.mask)
    return batch_like(G, v_nd, skip_masking=True)


def d(G: Batch) -> Batch:
    # get diagonals
    assert G.order == 2
    v_d = get_diag(G.indices, G.values, G.n_nodes, G.mask, G.node_mask, G.node_ofs)
    return Batch(None, v_d, G.n_nodes, None, G.node_mask, True, G.t_indices, G.t_mask, G.node_mask, G.node_ofs)


def v2d(G: Batch, values: Batch) -> Batch:
    # vectors to diagonal matrices
    v_d = to_diag(G.indices, values.values, G.mask, G.node_mask)
    return batch_like(G, v_d, skip_masking=True)


def apply(G: Union[torch.Tensor, Batch], f: Callable[[torch.Tensor], torch.Tensor], skip_masking=False) -> Union[torch.Tensor, Batch]:
    if isinstance(G, torch.Tensor):
        return f(G)
    return batch_like(G, f(G.values), skip_masking)


def add_batch(G1: Union[torch.Tensor, Batch], G2: Union[torch.Tensor, Batch]) -> Union[torch.Tensor, Batch]:
    # add features of two batched graphs with identical edge structures
    if isinstance(G1, Batch) and isinstance(G2, Batch):
        assert G1.order == G2.order
        assert G1.n_nodes == G2.n_nodes
        assert G1.n_edges == G2.n_edges
        return batch_like(G1, G1.values + G2.values, skip_masking=True)
    else:
        assert isinstance(G1, torch.Tensor) and isinstance(G2, torch.Tensor)
        assert G1.size() == G2.size()
        return G1 + G2


def make_batch_concatenated(node_feature: torch.Tensor, edge_index: torch.LongTensor, edge_feature: torch.Tensor,
                            n_nodes: list, n_edges: list) -> Batch:
    """
    :param node_feature: Tensor([sum(n), Dv])
    :param edge_index: LongTensor([2, sum(e)])
    :param edge_feature: Tensor([sum(e), De])
    :param n_nodes: list
    :param n_edges: list
    """
    assert len(node_feature.size()) == len(edge_index.size()) == len(edge_feature.size()) == 2
    bsize = len(n_nodes)
    n = node_feature.size(0)
    e = edge_feature.size(0)
    node_dim = node_feature.size(-1)
    edge_dim = edge_feature.size(-1)
    device = node_feature.device
    dtype = node_feature.dtype
    # unpack nodes
    idx = torch.arange(max(n_nodes), device=device)
    idx = idx[None, :].expand(bsize, max(n_nodes))  # [B, N]
    node_index = torch.arange(max(n_nodes), device=device, dtype=torch.long)
    node_index = node_index[None, :, None].expand(bsize, max(n_nodes), 2)  # [B, N, 2]
    node_num_vec = torch.tensor(n_nodes, device=device)[:, None]  # [B, 1]
    unpacked_node_index = node_index[idx < node_num_vec]  # [N, 2]
    unpacked_node_feature = torch.cat([node_feature, torch.zeros(n, edge_dim, device=device, dtype=dtype)], -1)
    # unpack edges
    edge_num_vec = torch.tensor(n_edges, device=device)[:, None]  # [B, 1]
    unpacked_edge_index = edge_index.t()  # [|E|, 2]
    unpacked_edge_feature = torch.cat([torch.zeros(e, node_dim, device=device, dtype=dtype), edge_feature], -1)
    # compose tensor
    n_edges_ = [n + e for n, e in zip(n_nodes, n_edges)]
    max_size = max(n_edges_)
    edge_index_ = torch.zeros(bsize, max_size, 2, device=device, dtype=torch.long)  # [B, N + |E|, 2]
    edge_feature_ = torch.zeros(bsize, max_size, node_dim + edge_dim, device=device, dtype=dtype)  # [B, N + |E|, D]
    full_index = torch.arange(max_size, device=device)[None, :].expand(bsize, max_size)  # [B, N + |E|]
    node_mask = full_index < node_num_vec  # [B, N + |E|]
    edge_mask = (node_num_vec <= full_index) & (full_index < node_num_vec + edge_num_vec)  # [B, N + |E|]
    edge_index_[node_mask] = unpacked_node_index
    edge_index_[edge_mask] = unpacked_edge_index
    edge_feature_[node_mask] = unpacked_node_feature
    edge_feature_[edge_mask] = unpacked_edge_feature
    # setup batch
    return Batch(edge_index_, edge_feature_, n_nodes, n_edges_)


def make_batch(node_features: list, edge_indices: list, edge_features: list) -> Batch:
    """interface for sparse batch construction
    :param node_features: List([Tensor([n, Dv])])
    :param edge_indices: List([LongTensor([2, e])])
    :param edge_features: List(Tensor([e, De]))
    """
    node_feature = torch.cat(node_features)
    edge_index = torch.cat(edge_indices, dim=1)
    edge_feature = torch.cat(edge_features)
    n_nodes = [x.size(0) for x in node_features]
    n_edges = [e.size(0) for e in edge_features]
    return make_batch_concatenated(node_feature, edge_index, edge_feature, n_nodes, n_edges)


def _make_batch(node_features: list, edge_indices: list, edge_features: list) -> Batch:
    """same to make_batch(), but slower
    :param node_features: List([Tensor([n, Dv])])
    :param edge_indices: List([LongTensor([2, e])])
    :param edge_features: List(Tensor([e, De]))
    """
    bsize = len(node_features)
    device = node_features[0].device
    n_nodes = [x.size(0) for x in node_features]
    n_edges = [e.size(0) for e in edge_features]
    node_dim = node_features[0].size(1)
    edge_dim = edge_features[0].size(1)
    dtype = node_features[0].dtype
    indices = torch.zeros(bsize, max(n_nodes) + max(n_edges), 2, device=device, dtype=torch.long)  # [B, N + |E|, 2]
    values = torch.zeros(bsize, max(n_nodes) + max(n_edges), node_dim + edge_dim, device=device, dtype=dtype)  # [B, N + |E|, Dv+De]
    for idx, tup in enumerate(zip(node_features, edge_indices, edge_features)):
        node_feat, edge_idx, edge_feat = tup
        n = node_feat.size(0)
        e = edge_idx.size(1)
        assert (edge_idx[0, :] != edge_idx[1, :]).all()
        indices[idx, :n, :] = torch.arange(n)[:, None].expand(n, 2)  # node indices
        indices[idx, n:n + e, :] = edge_idx.t()  # edge indices
        values[idx, :n, :node_dim] = node_feat
        values[idx, n:n + e, node_dim:] = edge_feat
    # setup batch
    n_edges = [n + e for n, e in zip(n_nodes, n_edges)]
    return Batch(indices, values, n_nodes, n_edges)
