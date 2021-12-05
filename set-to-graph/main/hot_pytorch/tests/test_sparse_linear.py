import time

import torch

from ..models.sparse.masksum import batch_mask, diagonalization_mask as diag_mask, count_unique, loop_exclusion_mask as loop_mask
from ..models.sparse import Linear, SumPool, AvgPool, MaxPool
from ..batch.sparse import Batch, make_batch
from ..utils.sparse import to_diag, get_nondiag, get_diag, get_transpose_info, do_transpose


def test_unique():
    mat = torch.tensor([[1, 2, 3, 4],
                        [1, 1, 2, 3],
                        [2, 4, 2, 6],
                        [0, 1, 0, -1],
                        [0, 0, 0, 0]], dtype=torch.long)
    assert (count_unique(mat) == torch.tensor([4, 3, 3, 2, 1])).all()


def test_loop_mask():
    idxs = torch.tensor([[1, 2, 3, 4],
                         [1, 2, 3, -1],
                         [1, 2, -1, -1]], dtype=torch.long)
    row_idxs = idxs.unsqueeze(-1)
    col_idxs = idxs.unsqueeze(-1)
    mask = loop_mask(row_idxs, col_idxs)
    assert (mask[0] == 1 - torch.eye(4)).all()
    assert (mask[1, :3, :3] == 1 - torch.eye(3)).all()
    assert (mask[2, :2, :2] == 1 - torch.eye(2)).all()


def test_1_0():
    n_nodes = [4, 1, 2, 2]
    G = Batch(None, torch.ones(4, 4, 5), n_nodes, None, None)

    V1 = G.values.sum(1)
    assert ((V1[..., 0] - torch.tensor([4, 1, 2, 2])).abs() < 1e-4).all()

    n_vec = torch.tensor(G.n_nodes).to(V1).float().unsqueeze(-1)  # [B, 1]
    n_vec = n_vec.masked_fill_(n_vec == 0, 1e-5)
    V1 = V1 / n_vec  # [B, D]
    assert ((V1[..., 0] - 1).abs() < 1e-4).all()


def test_1_1():
    n_nodes = [4, 1, 2, 2]
    G = Batch(None, torch.ones(4, 4, 5), n_nodes, None, None)

    V1 = G.values.clone()

    n_vec = (torch.tensor(G.n_nodes) - 1).to(V1).float().view(4, 1, 1)  # [B, 1, 1]
    n_vec = n_vec.masked_fill_(n_vec == 0, 1e-5)
    values = G.values / n_vec  # [B, D]
    M2 = (torch.ones(4, 4) - torch.eye(4))
    V2 = torch.einsum('ji,bid->bjd', M2, values)
    assert ((V1[0, :, 0] - 1).abs() < 1e-4).all()
    assert ((V2[0, :, 0] - 1).abs() < 1e-4).all()
    assert ((V1[2, :2, 0] - 1).abs() < 1e-4).all()
    assert ((V2[2, :2, 0] - 1).abs() < 1e-4).all()

    n_nodes = [4, 1, 2, 2]
    node_feat = torch.tensor([[1, 2, 3, 4],
                              [5, 0, 0, 0],
                              [6, 7, 0, 0],
                              [8, 9, 0, 0]])[:, :, None].repeat(1, 1, 5).float()
    G = Batch(None, node_feat, n_nodes, None, None)

    V1 = G.values.clone()

    n_vec = (torch.tensor(G.n_nodes) - 1).to(V1).float().view(4, 1, 1)  # [B, 1, 1]
    n_vec = n_vec.masked_fill_(n_vec == 0, 1e-5)
    values = G.values / n_vec  # [B, D]
    M2 = (torch.ones(4, 4) - torch.eye(4))
    V2 = torch.einsum('ji,bid->bjd', M2, values)
    assert ((V1[0, :, 0] - torch.tensor([1, 2, 3, 4])).abs() < 1e-4).all()
    assert ((V2[0, :, 0] - torch.tensor([9/3, 8/3, 7/3, 6/3])).abs() < 1e-4).all()
    assert ((V1[2, :2, 0] - torch.tensor([6, 7])).abs() < 1e-4).all()
    assert ((V2[2, :2, 0] - torch.tensor([7, 6])).abs() < 1e-4).all()


def test_2_0():
    A1_nodes = torch.tensor([1, 1, 1, 1]).float()
    A1_edges = torch.tensor([[0, 1, 0, 1],
                             [1, 0, 1, 1],
                             [0, 1, 0, 1],
                             [1, 0, 1, 0]]).float()
    A2_nodes = torch.tensor([1]).float()
    A2_edges = torch.tensor([[0]]).float()
    A3_nodes = torch.tensor([1, 1]).float()
    A3_edges = torch.tensor([[0, 0],
                             [0, 0]]).float()
    A4_nodes = torch.tensor([1, 1]).float()
    A4_edges = torch.tensor([[0, 0],
                             [1, 0]]).float()

    node_list = [A1_nodes, A2_nodes, A3_nodes, A4_nodes]
    node_features = [x[:, None].repeat(1, 4) for x in node_list]
    edge_list = [A1_edges, A2_edges, A3_edges, A4_edges]
    edge_features = [e.to_sparse().coalesce() for e in edge_list]
    edge_indices = [e.indices() for e in edge_features]
    edge_features = [e.values()[:, None].repeat(1, 4) for e in edge_features]
    G = make_batch(node_features, edge_indices, edge_features)

    indices, values = G.indices, G.values
    V1 = get_diag(indices, values, G.n_nodes, G.mask, G.node_mask, G.node_ofs).sum(1)
    V2 = get_nondiag(indices, values, G.mask).sum(1)

    n_vec = torch.tensor(G.n_nodes).unsqueeze(-1).to(V1).float()  # [B, 1]
    n_vec = n_vec.masked_fill_(n_vec == 0, 1e-5)
    V1 = V1 / n_vec  # [B, D]
    e_vec = (torch.tensor(G.n_edges) - torch.tensor(G.n_nodes)).unsqueeze(-1).to(V1).float()  # [B, 1]
    e_vec = e_vec.masked_fill_(e_vec == 0, 1e-5)
    V2 = V2 / e_vec  # [B, D]

    assert ((V1[:, 0] - 1).abs() < 1e-4).all()
    assert ((V2[:, 4] - torch.tensor([1, 0, 0, 1])).abs() < 1e-4).all()


def _normalize(values, mask):
    vec = mask.sum(-1, keepdim=True)
    return values / vec.masked_fill_(vec == 0, 1e-5)


def test_2_1():
    A1_nodes = torch.tensor([2, 2, 2, 2]).float()
    A1_edges = torch.tensor([[0, 1, 0, 1],
                             [1, 0, 1, 1],
                             [0, 1, 0, 1],
                             [1, 0, 1, 0]]).float()
    A2_nodes = torch.tensor([2]).float()
    A2_edges = torch.tensor([[0]]).float()
    A3_nodes = torch.tensor([2, 2]).float()
    A3_edges = torch.tensor([[0, 0],
                             [0, 0]]).float()
    A4_nodes = torch.tensor([2, 2]).float()
    A4_edges = torch.tensor([[0, 0],
                             [1, 0]]).float()

    node_list = [A1_nodes, A2_nodes, A3_nodes, A4_nodes]
    node_features = [x[:, None].repeat(1, 4) for x in node_list]
    edge_list = [A1_edges, A2_edges, A3_edges, A4_edges]
    edge_features = [e.to_sparse().coalesce() for e in edge_list]
    edge_indices = [e.indices() for e in edge_features]
    edge_features = [e.values()[:, None].repeat(1, 4) for e in edge_features]
    G = make_batch(node_features, edge_indices, edge_features)

    indices, values = G.indices, G.values
    bsize = indices.size(0)
    n = max(G.n_nodes)

    diagonal = get_diag(indices, values, G.n_nodes, G.mask, G.node_mask, G.node_ofs)
    V1 = diagonal.clone()
    assert ((V1[..., 0] - torch.tensor([[2, 2, 2, 2],
                                        [2, 0, 0, 0],
                                        [2, 2, 0, 0],
                                        [2, 2, 0, 0]])).abs() < 1e-4).all()

    M = batch_mask(G.node_mask, G.mask)
    node_idx = torch.arange(n).view(1, n, 1).expand(bsize, n, 1)
    row_idx = indices[..., :1]
    col_idx = indices[..., 1:]

    M2 = (M & loop_mask(node_idx, col_idx) & diag_mask(node_idx, row_idx)).float()
    V2 = torch.einsum('bji,bid->bjd', M2, values)
    assert ((V2[..., 4] - torch.tensor([[2, 3, 2, 2],
                                        [0, 0, 0, 0],
                                        [0, 0, 0, 0],
                                        [0, 1, 0, 0]])).abs() < 1e-4).all()
    V2 = _normalize(V2, M2)
    assert ((V2[..., 4] - torch.tensor([[1, 1, 1, 1],
                                        [0, 0, 0, 0],
                                        [0, 0, 0, 0],
                                        [0, 1, 0, 0]])).abs() < 1e-4).all()

    M3 = (M & loop_mask(node_idx, row_idx) & diag_mask(node_idx, col_idx)).float()
    V3 = torch.einsum('bji,bid->bjd', M3, values)
    assert ((V3[..., 4] - torch.tensor([[2, 2, 2, 3],
                                        [0, 0, 0, 0],
                                        [0, 0, 0, 0],
                                        [1, 0, 0, 0]])).abs() < 1e-4).all()
    V3 = _normalize(V3, M3)
    assert ((V3[..., 4] - torch.tensor([[1, 1, 1, 1],
                                        [0, 0, 0, 0],
                                        [0, 0, 0, 0],
                                        [1, 0, 0, 0]])).abs() < 1e-4).all()

    n_vec = (torch.tensor(G.n_nodes) - 1).view(bsize, 1, 1).to(V1).float()
    n_vec = n_vec.masked_fill_(n_vec == 0, 1e-5)
    diagonal_norm = diagonal / n_vec  # [B, N, D]
    M4 = (torch.ones(n, n) - torch.eye(n)).to(values.device)
    V4 = torch.einsum('ji,bid->bjd', M4, diagonal_norm)
    V4 = V4.masked_fill_(~G.node_mask.unsqueeze(-1), 0)
    assert ((V4[..., 0] - torch.tensor([[2, 2, 2, 2],
                                        [0, 0, 0, 0],
                                        [2, 2, 0, 0],
                                        [2, 2, 0, 0]])).abs() < 1e-4).all()

    M5 = (M & loop_mask(node_idx, indices)).float()
    V5 = torch.einsum('bji,bid->bjd', M5, values)
    assert ((V5[..., 4] - torch.tensor([[5, 4, 5, 4],
                                        [0, 0, 0, 0],
                                        [0, 0, 0, 0],
                                        [0, 0, 0, 0]])).abs() < 1e-4).all()
    V5 = _normalize(V5, M5)
    assert ((V5[..., 4] - torch.tensor([[1, 1, 1, 1],
                                        [0, 0, 0, 0],
                                        [0, 0, 0, 0],
                                        [0, 0, 0, 0]])).abs() < 1e-4).all()


def test_2_2():
    A1_nodes = torch.tensor([1, 2, 3, 4]).float()
    A1_edges = torch.tensor([[0, 1, 0, 1],
                             [1, 0, 1, 1],
                             [0, 1, 0, 1],
                             [1, 0, 1, 0]]).float()
    A2_nodes = torch.tensor([2]).float()
    A2_edges = torch.tensor([[0]]).float()
    A3_nodes = torch.tensor([1, 2]).float()
    A3_edges = torch.tensor([[0, 0],
                             [0, 0]]).float()
    A4_nodes = torch.tensor([1, 2]).float()
    A4_edges = torch.tensor([[0, 0],
                             [1, 0]]).float()

    node_list = [A1_nodes, A2_nodes, A3_nodes, A4_nodes]
    node_features = [x[:, None].repeat(1, 4) for x in node_list]
    edge_list = [A1_edges, A2_edges, A3_edges, A4_edges]
    edge_features = [e.to_sparse().coalesce() for e in edge_list]
    edge_indices = [e.indices() for e in edge_features]
    edge_features = [e.values()[:, None].repeat(1, 4) for e in edge_features]
    G = make_batch(node_features, edge_indices, edge_features)

    indices, values = G.indices, G.values
    bsize = indices.size(0)

    diagonal = get_diag(indices, values, G.n_nodes, G.mask, G.node_mask, G.node_ofs)
    V1 = to_diag(indices, diagonal, G.mask, G.node_mask)
    assert ((V1[..., 0] - torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])).abs() < 1e-4).all()

    transpose_indices, exist_mask = get_transpose_info(indices, G.mask)
    values_t = do_transpose(values, transpose_indices, exist_mask)
    V2 = get_nondiag(indices, values, G.mask)
    assert ((V2[..., 4] - torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])).abs() < 1e-4).all()

    V3 = get_nondiag(indices, values_t, G.mask)
    assert ((V3[..., 4] - torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])).abs() < 1e-4).all()

    row_idx = indices[..., :1]
    col_idx = indices[..., 1:]
    dim = values.size(-1)

    V4 = torch.gather(diagonal, 1, row_idx.clone().clamp_(min=0).repeat(1, 1, dim))
    V4 = get_nondiag(indices, V4, G.mask)
    assert ((V4[..., 0] - torch.tensor([[0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])).abs() < 1e-4).all()

    V5 = torch.gather(diagonal, 1, col_idx.clone().clamp_(min=0).repeat(1, 1, dim))
    V5 = get_nondiag(indices, V5, G.mask)
    assert ((V5[..., 0] - torch.tensor([[0, 0, 0, 0, 2, 4, 1, 3, 4, 2, 4, 1, 3],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])).abs() < 1e-4).all()

    n = max(G.n_nodes)

    M_ee = batch_mask(G.mask, G.mask)
    M_ne = batch_mask(G.node_mask, G.mask)
    M_en = batch_mask(G.mask, G.node_mask)

    node_idx = torch.arange(n).view(1, n, 1).expand(bsize, n, 1).to(values.device)
    row_idx = indices[..., :1]
    col_idx = indices[..., 1:]

    M6 = (M_ne & loop_mask(node_idx, col_idx) & diag_mask(node_idx, row_idx)).float()
    V6 = M6.bmm(values)
    V6 = _normalize(V6, M6)
    V6 = to_diag(indices, V6, G.mask, G.node_mask)
    assert ((V6[..., 4] - torch.tensor([[2/2, 3/3, 2/2, 2/2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 1/1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])).abs() < 1e-4).all()

    M7 = (M_ne & loop_mask(node_idx, row_idx) & diag_mask(node_idx, col_idx)).float()
    V7 = M7.bmm(values)
    V7 = _normalize(V7, M7)
    V7 = to_diag(indices, V7, G.mask, G.node_mask)
    assert ((V7[..., 4] - torch.tensor([[2/2, 2/2, 2/2, 3/3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [1/1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])).abs() < 1e-4).all()

    # Oij = sum_{k!=i,j} Akj
    M8 = (M_ee & loop_mask(indices, row_idx) & diag_mask(col_idx, col_idx)).float()
    V8 = M8.bmm(values)
    assert ((V8[0, :, 4] - torch.tensor([0, 0, 0, 0, 1, 2, 1, 1, 2, 1, 2, 1, 1])).abs() < 1e-4).all()

    # Oij = sum_{k!=i,j} Aik
    M9 = (M_ee & loop_mask(indices, col_idx) & diag_mask(row_idx, row_idx)).float()
    V9 = M9.bmm(values)
    assert ((V9[0, :, 4] - torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 2, 1, 1, 1, 1])).abs() < 1e-4).all()

    # Oij = sum_{k!=i,j} Ajk
    M10 = (M_ee & loop_mask(indices, col_idx) & diag_mask(col_idx, row_idx)).float()
    V10 = M10.bmm(values)
    assert ((V10[0, :, 4] - torch.tensor([0, 0, 0, 0, 2, 1, 1, 1, 2, 2, 1, 1, 1])).abs() < 1e-4).all()

    # Oij = sum_{k!=i,j} Aki
    M11 = (M_ee & loop_mask(indices, row_idx) & diag_mask(row_idx, col_idx)).float()
    V11 = M11.bmm(values)
    assert ((V11[0, :, 4] - torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 1, 1, 2, 2])).abs() < 1e-4).all()

    V8 = _normalize(V8, M8)
    V9 = _normalize(V9, M9)
    V10 = _normalize(V10, M10)
    V11 = _normalize(V11, M11)
    assert ((V8[0, :, 4] - torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])).abs() < 1e-4).all()
    assert ((V9[0, :, 4] - torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])).abs() < 1e-4).all()
    assert ((V10[0, :, 4] - torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])).abs() < 1e-4).all()
    assert ((V11[0, :, 4] - torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])).abs() < 1e-4).all()

    n_vec = (torch.tensor(G.n_nodes) - 1).view(4, 1, 1).to(V1).float()  # [B, 1]
    n_vec = n_vec.masked_fill_(n_vec == 0, 1e-5)
    diagonal_norm = diagonal / n_vec  # [B, D]
    M12 = (torch.ones(n, n) - torch.eye(n)).to(values.device)
    V12 = torch.einsum('ji,bid->bjd', M12, diagonal_norm)
    V12 = to_diag(indices, V12, G.mask, G.node_mask)
    assert ((V12[..., 0] - torch.tensor([[9/3, 8/3, 7/3, 6/3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [2/1, 1/1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [2/1, 1/1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])).abs() < 1e-4).all()

    M13 = (M_ne & loop_mask(node_idx, indices)).float()
    V13 = M13.bmm(values)
    V13 = _normalize(V13, M13)
    assert ((V13[0, :, 4] - torch.tensor([5/5, 4/4, 5/5, 4/4])).abs() < 1e-4).all()
    V13 = to_diag(indices, V13, G.mask, G.node_mask)

    M14 = (M_en & loop_mask(indices, node_idx)).float()
    V14 = M14.bmm(diagonal)
    assert ((V14[0, :, 0] - torch.tensor([0, 0, 0, 0, 7, 5, 7, 5, 4, 5, 3, 5, 3])).abs() < 1e-4).all()

    M15 = (M_ee & loop_mask(indices, indices)).float()
    V15 = M15.bmm(values)
    assert ((V15[0, :, 4] - torch.tensor([0, 0, 0, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2])).abs() < 1e-4).all()

    V14 = _normalize(V14, M14)
    V15 = _normalize(V15, M15)
    assert ((V14[0, :, 0] - torch.tensor([0, 0, 0, 0, 7/2, 5/2, 7/2, 5/2, 4/2, 5/2, 3/2, 5/2, 3/2])).abs() < 1e-4).all()
    assert ((V15[0, :, 4] - torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1])).abs() < 1e-4).all()


def test_forward():
    x = torch.ones(8, 100, 10)
    n_nodes = [40, 60, 80, 100, 40, 60, 80, 100]
    G1 = Batch(None, x, n_nodes, None)

    n_nodes = [40, 60, 80, 100, 40, 60, 80, 100]
    node_list = [torch.ones(n) for n in n_nodes]
    node_features = [x[:, None].repeat(1, 5) for x in node_list]
    edge_list = [(torch.rand(n, n) - torch.eye(n) > 0.75).float() for n in n_nodes]
    edge_features = [e.to_sparse().coalesce() for e in edge_list]
    edge_indices = [e.indices() for e in edge_features]
    edge_features = [e.values()[:, None].repeat(1, 5) for e in edge_features]
    G2 = make_batch(node_features, edge_indices, edge_features)
    N_e = G2.values.size(1)

    device = 'cuda'
    G1.to(device)
    G2.to(device)

    print('1->0')
    l1 = Linear(1, 0, 10, 20, True, 'default', False).to(device)
    l2 = Linear(1, 0, 10, 20, True, 'default', True).to(device)
    l3 = Linear(1, 0, 10, 20, True, 'light', False).to(device)
    tic = time.time()
    assert l1(G1).shape == torch.Size([8, 20])
    print(time.time() - tic)
    tic = time.time()
    assert l2(G1).shape == torch.Size([8, 20])
    print(time.time() - tic)
    tic = time.time()
    assert l3(G1).shape == torch.Size([8, 20])
    print(time.time() - tic)

    print('1->1')
    l1 = Linear(1, 1, 10, 20, True, 'default', False).to(device)
    l2 = Linear(1, 1, 10, 20, True, 'default', True).to(device)
    l3 = Linear(1, 1, 10, 20, True, 'light', False).to(device)
    tic = time.time()
    assert l1(G1).values.shape == torch.Size([8, 100, 20])
    print(time.time() - tic)
    tic = time.time()
    assert l2(G1).values.shape == torch.Size([8, 100, 20])
    print(time.time() - tic)
    tic = time.time()
    assert l3(G1).values.shape == torch.Size([8, 100, 20])
    print(time.time() - tic)

    print('2->0')
    l1 = Linear(2, 0, 10, 20, True, 'default', False).to(device)
    l2 = Linear(2, 0, 10, 20, True, 'default', True).to(device)
    l3 = Linear(2, 0, 10, 20, True, 'light', False).to(device)
    tic = time.time()
    assert l1(G2).shape == torch.Size([8, 20])
    print(time.time() - tic)
    tic = time.time()
    assert l2(G2).shape == torch.Size([8, 20])
    print(time.time() - tic)
    tic = time.time()
    assert l3(G2).shape == torch.Size([8, 20])
    print(time.time() - tic)

    print('2->1')
    l1 = Linear(2, 1, 10, 20, True, 'default', False).to(device)
    l2 = Linear(2, 1, 10, 20, True, 'default', True).to(device)
    l3 = Linear(2, 1, 10, 20, True, 'light', False).to(device)
    tic = time.time()
    assert l1(G2).values.shape == torch.Size([8, 100, 20])
    print(time.time() - tic)
    tic = time.time()
    assert l2(G2).values.shape == torch.Size([8, 100, 20])
    print(time.time() - tic)
    tic = time.time()
    assert l3(G2).values.shape == torch.Size([8, 100, 20])
    print(time.time() - tic)

    print('2->2')
    l1 = Linear(2, 2, 10, 20, True, 'default', False).to(device)
    l2 = Linear(2, 2, 10, 20, True, 'default', True).to(device)
    l3 = Linear(2, 2, 10, 20, True, 'light', False).to(device)
    tic = time.time()
    assert l1(G2).values.shape == torch.Size([8, N_e, 20])
    print(time.time() - tic)
    tic = time.time()
    assert l2(G2).values.shape == torch.Size([8, N_e, 20])
    print(time.time() - tic)
    tic = time.time()
    assert l3(G2).values.shape == torch.Size([8, N_e, 20])
    print(time.time() - tic)


def test_pool():
    A1_nodes = torch.tensor([1]).float()
    A1_edges = (torch.rand(1, 1) - torch.eye(1) > 0.25).float()
    A2_nodes = torch.tensor([1, 1]).float()
    A2_edges = (torch.rand(2, 2) - torch.eye(2) > 0.25).float()
    A3_nodes = torch.tensor([1, 1, 1]).float()
    A3_edges = (torch.rand(3, 3) - torch.eye(3) > 0.25).float()
    A4_nodes = -torch.tensor([4, 7, 5, 2]).float()
    A4_edges = -torch.tensor([[0, 0, 3, 4],
                              [3, 0, 1, 0],
                              [0, 4, 0, 2],
                              [1, 2, 0, 0]]).float()
    A5_nodes = torch.tensor([1, 1])
    A5_edges = (torch.rand(2, 2) - torch.eye(2) > 0.25).float()

    node_list = [A1_nodes, A2_nodes, A3_nodes, A4_nodes, A5_nodes]
    node_features = [x[:, None].repeat(1, 5) for x in node_list]
    edge_list = [A1_edges, A2_edges, A3_edges, A4_edges, A5_edges]
    edge_features = [e.to_sparse().coalesce() for e in edge_list]
    edge_indices = [e.indices() for e in edge_features]
    edge_features = [e.values()[:, None].repeat(1, 5) for e in edge_features]
    G = make_batch(node_features, edge_indices, edge_features)

    assert (MaxPool(2)(G)[3, 0] + 2).abs() < 1e-4
    assert (MaxPool(2)(G)[3, 5] + 1).abs() < 1e-4

    A4_nodes = torch.tensor([1, 1, 1, 1]).float()
    A4_edges = torch.tensor([[0, 1, 1, 1],
                             [1, 0, 1, 1],
                             [1, 1, 0, 1],
                             [1, 1, 1, 0]]).float()

    node_list = [A1_nodes, A2_nodes, A3_nodes, A4_nodes, A5_nodes]
    node_features = [x[:, None].repeat(1, 5) for x in node_list]
    edge_list = [A1_edges, A2_edges, A3_edges, A4_edges, A5_edges]
    edge_features = [e.to_sparse().coalesce() for e in edge_list]
    edge_indices = [e.indices() for e in edge_features]
    edge_features = [e.values()[:, None].repeat(1, 5) for e in edge_features]
    G = make_batch(node_features, edge_indices, edge_features)

    assert (SumPool(2)(G)[3, 0] - 4).abs() < 1e-4
    assert (SumPool(2)(G)[3, 5] - 12).abs() < 1e-4

    A4_nodes = torch.tensor([2, 2, 2, 2]).float()
    A4_edges = torch.tensor([[0, 1, 1, 1],
                             [1, 0, 1, 0],
                             [1, 0, 0, 1],
                             [1, 1, 1, 0]]).float()

    node_list = [A1_nodes, A2_nodes, A3_nodes, A4_nodes, A5_nodes]
    node_features = [x[:, None].repeat(1, 5) for x in node_list]
    edge_list = [A1_edges, A2_edges, A3_edges, A4_edges, A5_edges]
    edge_features = [e.to_sparse().coalesce() for e in edge_list]
    edge_indices = [e.indices() for e in edge_features]
    edge_features = [e.values()[:, None].repeat(1, 5) for e in edge_features]
    G = make_batch(node_features, edge_indices, edge_features)

    assert (AvgPool(2)(G)[3, 0] - 2).abs() < 1e-4
    assert (AvgPool(2)(G)[3, 5] - 1).abs() < 1e-4


def test_backward():
    x = torch.ones(8, 100, 10)
    n_nodes = [40, 60, 80, 100, 40, 60, 80, 100]
    G1 = Batch(None, x, n_nodes, None)

    n_nodes = [40, 60, 80, 100, 40, 60, 80, 100]
    node_list = [torch.ones(n) for n in n_nodes]
    node_features = [x[:, None].repeat(1, 5) for x in node_list]
    edge_list = [(torch.rand(n, n) - torch.eye(n) > 0.75).float() for n in n_nodes]
    edge_features = [e.to_sparse().coalesce() for e in edge_list]
    edge_indices = [e.indices() for e in edge_features]
    edge_features = [e.values()[:, None].repeat(1, 5) for e in edge_features]
    G2 = make_batch(node_features, edge_indices, edge_features)
    N_e = G2.values.size(1)
    print(N_e)

    device = 'cuda'
    G1.to(device)
    G2.to(device)
    p1 = AvgPool(1).to(device)
    p2 = AvgPool(2).to(device)

    def assert_grad(layer):
        w = layer.weight.weight
        assert not w.grad.isnan().any() and not w.grad.isinf().any()
        for b in layer.bias.bias:
            assert not b.grad.isnan().any() and not b.grad.isinf().any()

    print('1->0')
    l1 = Linear(1, 0, 10, 20, True, 'default', False).to(device)
    l2 = Linear(1, 0, 10, 20, True, 'default', True).to(device)
    l3 = Linear(1, 0, 10, 20, True, 'light', False).to(device)
    scalar = l1(G1).sum()
    tic = time.time()
    scalar.backward()
    print(time.time() - tic)
    assert_grad(l1)
    scalar = l2(G1).sum()
    tic = time.time()
    scalar.backward()
    print(time.time() - tic)
    assert_grad(l2)
    scalar = l3(G1).sum()
    tic = time.time()
    scalar.backward()
    print(time.time() - tic)
    assert_grad(l3)

    print('1->1')
    l1 = Linear(1, 1, 10, 20, True, 'default', False).to(device)
    l2 = Linear(1, 1, 10, 20, True, 'default', True).to(device)
    l3 = Linear(1, 1, 10, 20, True, 'light', False).to(device)
    scalar = p1(l1(G1)).sum()
    tic = time.time()
    scalar.backward()
    print(time.time() - tic)
    assert_grad(l1)
    scalar = p1(l2(G1)).sum()
    tic = time.time()
    scalar.backward()
    print(time.time() - tic)
    assert_grad(l2)
    scalar = p1(l3(G1)).sum()
    tic = time.time()
    scalar.backward()
    print(time.time() - tic)
    assert_grad(l3)

    print('2->0')
    l1 = Linear(2, 0, 10, 20, True, 'default', False).to(device)
    l2 = Linear(2, 0, 10, 20, True, 'default', True).to(device)
    l3 = Linear(2, 0, 10, 20, True, 'light', False).to(device)
    scalar = l1(G2).sum()
    tic = time.time()
    scalar.backward()
    print(time.time() - tic)
    assert_grad(l1)
    scalar = l2(G2).sum()
    tic = time.time()
    scalar.backward()
    print(time.time() - tic)
    assert_grad(l2)
    scalar = l3(G2).sum()
    tic = time.time()
    scalar.backward()
    print(time.time() - tic)
    assert_grad(l3)

    print('2->1')
    l1 = Linear(2, 1, 10, 20, True, 'default', False).to(device)
    l2 = Linear(2, 1, 10, 20, True, 'default', True).to(device)
    l3 = Linear(2, 1, 10, 20, True, 'light', False).to(device)
    scalar = p1(l1(G2)).sum()
    tic = time.time()
    scalar.backward()
    print(time.time() - tic)
    assert_grad(l1)
    scalar = p1(l2(G2)).sum()
    tic = time.time()
    scalar.backward()
    print(time.time() - tic)
    assert_grad(l2)
    scalar = p1(l3(G2)).sum()
    tic = time.time()
    scalar.backward()
    print(time.time() - tic)
    assert_grad(l3)

    print('2->2')
    l1 = Linear(2, 2, 10, 20, True, 'default', False).to(device)
    l2 = Linear(2, 2, 10, 20, True, 'default', True).to(device)
    l3 = Linear(2, 2, 10, 20, True, 'light', False).to(device)
    scalar = p2(l1(G2)).sum()
    tic = time.time()
    scalar.backward()
    print(time.time() - tic)
    assert_grad(l1)
    scalar = p2(l2(G2)).sum()
    tic = time.time()
    scalar.backward()
    print(time.time() - tic)
    assert_grad(l2)
    scalar = p2(l3(G2)).sum()
    tic = time.time()
    scalar.backward()
    print(time.time() - tic)
    assert_grad(l3)
