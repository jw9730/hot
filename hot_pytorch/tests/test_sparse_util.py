import torch

from ..batch.sparse import make_batch, t, v2d, nd, d


def test_make_batch():
    A1_nodes = torch.tensor([1, 0, 1, 0])
    A1_edges = torch.tensor([[0, 1, 0, 1],
                             [1, 0, 1, 1],
                             [0, 1, 0, 1],
                             [1, 0, 1, 0]])
    A2_nodes = torch.tensor([1])
    A2_edges = torch.tensor([[0]])
    A3_nodes = torch.tensor([1, 0])
    A3_edges = torch.tensor([[0, 1],
                             [1, 0]])
    A4_nodes = torch.tensor([1, 0])
    A4_edges = torch.tensor([[0, 0],
                             [1, 0]])

    node_list = [A1_nodes, A2_nodes, A3_nodes, A4_nodes]
    node_features = [x[:, None].repeat(1, 4) for x in node_list]
    edge_list = [A1_edges, A2_edges, A3_edges, A4_edges]
    edge_features = [e.to_sparse().coalesce() for e in edge_list]
    edge_indices = [e.indices() for e in edge_features]
    edge_features = [e.values()[:, None].repeat(1, 4) for e in edge_features]
    G = make_batch(node_features, edge_indices, edge_features)

    for i, v, m, n, x, e in zip(G.indices, G.values, G.mask, G.n_nodes, node_list, edge_list):
        A_dense = torch.sparse_coo_tensor(i[m].t(), v[m][..., 0], size=(n, n)).to_dense()
        assert (A_dense == x.diag()).all()
        A_dense = torch.sparse_coo_tensor(i[m].t(), v[m][..., 4], size=(n, n)).to_dense()
        assert (A_dense == e).all()


def test_transpose():
    A1_nodes = torch.tensor([1, 5, 9, 0])
    A1_edges = torch.tensor([[0, 2, 0, 3],
                             [4, 0, 6, 7],
                             [0, 8, 0, 8],
                             [7, 0, 6, 0]])
    A2_nodes = torch.tensor([1])
    A2_edges = torch.tensor([[0]])
    A3_nodes = torch.tensor([1, 0])
    A3_edges = torch.tensor([[0, 2],
                             [3, 0]])
    A4_nodes = torch.tensor([1, 0])
    A4_edges = torch.tensor([[0, 0],
                             [2, 0]])

    node_list = [A1_nodes, A2_nodes, A3_nodes, A4_nodes]
    node_features = [x[:, None].repeat(1, 4) for x in node_list]
    edge_list = [A1_edges, A2_edges, A3_edges, A4_edges]
    edge_features = [e.to_sparse().coalesce() for e in edge_list]
    edge_indices = [e.indices() for e in edge_features]
    edge_features = [e.values()[:, None].repeat(1, 4) for e in edge_features]
    G = make_batch(node_features, edge_indices, edge_features)

    G_T = t(G)

    A1_T_edges = torch.tensor([[0, 4, 0, 7],
                               [2, 0, 8, 0],
                               [0, 6, 0, 6],
                               [3, 0, 8, 0]])
    A2_T_edges = torch.tensor([[0]])
    A3_T_edges = torch.tensor([[0, 3],
                               [2, 0]])
    A4_T_edges = torch.tensor([[0, 0],
                               [0, 0]])
    edge_T_list = [A1_T_edges, A2_T_edges, A3_T_edges, A4_T_edges]

    for i, v, m, n, x, e in zip(G_T.indices, G_T.values, G_T.mask, G_T.n_nodes, node_list, edge_T_list):
        A_dense = torch.sparse_coo_tensor(i[m].t(), v[m][..., 0], size=(n, n)).to_dense()
        assert (A_dense == x.diag()).all()
        A_dense = torch.sparse_coo_tensor(i[m].t(), v[m][..., 4], size=(n, n)).to_dense()
        assert (A_dense == e).all()

    A1_nodes = torch.tensor([1, 0, 1, 0])
    A1_edges = torch.tensor([[0, 1, 0, 1],
                             [1, 0, 1, 1],
                             [0, 1, 0, 1],
                             [1, 0, 1, 0]])
    A2_nodes = torch.tensor([1])
    A2_edges = torch.tensor([[0]])
    A3_nodes = torch.tensor([1, 0])
    A3_edges = torch.tensor([[0, 1],
                             [1, 0]])
    A4_nodes = torch.tensor([1, 0])
    A4_edges = torch.tensor([[0, 0],
                             [1, 0]])

    node_list = [A1_nodes, A2_nodes, A3_nodes, A4_nodes]
    node_features = [x[:, None].repeat(1, 4) for x in node_list]
    edge_list = [A1_edges, A2_edges, A3_edges, A4_edges]
    edge_features = [e.to_sparse().coalesce() for e in edge_list]
    edge_indices = [e.indices() for e in edge_features]
    edge_features = [e.values()[:, None].repeat(1, 4) for e in edge_features]
    G = make_batch(node_features, edge_indices, edge_features)
    assert (t(t(G)).values == t(t(t(t(G)))).values).all()

    A1_nodes = torch.tensor([1, 0, 0, 0])
    A1_edges = torch.tensor([[0, 1, 0, 1],
                             [1, 0, 1, 1],
                             [0, 1, 0, 1],
                             [1, 1, 1, 0]])
    A2_nodes = torch.tensor([1])
    A2_edges = torch.tensor([[0]])
    A3_nodes = torch.tensor([1, 0])
    A3_edges = torch.tensor([[0, 1],
                             [1, 0]])
    A4_nodes = torch.tensor([1, 1])
    A4_edges = torch.tensor([[0, 0],
                             [0, 0]])

    node_list = [A1_nodes, A2_nodes, A3_nodes, A4_nodes]
    node_features = [x[:, None].repeat(1, 4) for x in node_list]
    edge_list = [A1_edges, A2_edges, A3_edges, A4_edges]
    edge_features = [e.to_sparse().coalesce() for e in edge_list]
    edge_indices = [e.indices() for e in edge_features]
    edge_features = [e.values()[:, None].repeat(1, 4) for e in edge_features]
    G = make_batch(node_features, edge_indices, edge_features)
    assert (G.values == t(G).values).all()


def test_diag():
    A1_nodes = torch.tensor([1, 2, 8, 3])
    A1_edges = torch.tensor([[0, 2, 0, 3],
                             [4, 0, 5, 6],
                             [0, 7, 0, 9],
                             [8, 0, 7, 0]])
    A2_nodes = torch.tensor([4])
    A2_edges = torch.tensor([[0]])
    A3_nodes = torch.tensor([7, 1])
    A3_edges = torch.tensor([[0, 2],
                             [3, 0]])
    A4_nodes = torch.tensor([5, 1])
    A4_edges = torch.tensor([[0, 0],
                             [2, 0]])

    node_list = [A1_nodes, A2_nodes, A3_nodes, A4_nodes]
    node_features = [x[:, None].repeat(1, 4) for x in node_list]
    edge_list = [A1_edges, A2_edges, A3_edges, A4_edges]
    edge_features = [e.to_sparse().coalesce() for e in edge_list]
    edge_indices = [e.indices() for e in edge_features]
    edge_features = [e.values()[:, None].repeat(1, 4) for e in edge_features]
    G = make_batch(node_features, edge_indices, edge_features)
    d_G = d(G)

    for v, m, n, x in zip(d_G.values, d_G.mask, d_G.n_nodes, node_list):
        assert (v[m][..., 0] == x).all()

    d_G_ = v2d(G, d_G)

    A1_nodes = torch.tensor([1, 2, 8, 3])
    A1_edges = torch.tensor([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]])
    A2_nodes = torch.tensor([4])
    A2_edges = torch.tensor([[0]])
    A3_nodes = torch.tensor([7, 1])
    A3_edges = torch.tensor([[0, 0],
                             [0, 0]])
    A4_nodes = torch.tensor([5, 1])
    A4_edges = torch.tensor([[0, 0],
                             [0, 0]])

    node_list = [A1_nodes, A2_nodes, A3_nodes, A4_nodes]
    edge_list = [A1_edges, A2_edges, A3_edges, A4_edges]

    for i, v, m, n, x, e in zip(d_G_.indices, d_G_.values, d_G_.mask, d_G_.n_nodes, node_list, edge_list):
        A_dense = torch.sparse_coo_tensor(i[m].t(), v[m][..., 4], size=(n, n)).to_dense()
        assert (A_dense == e).all()


def test_nondiag():
    A1_nodes = torch.tensor([1, 5, 9, 0])
    A1_edges = torch.tensor([[0, 2, 0, 3],
                             [4, 0, 6, 7],
                             [0, 8, 0, 8],
                             [7, 0, 6, 0]])
    A2_nodes = torch.tensor([1])
    A2_edges = torch.tensor([[0]])
    A3_nodes = torch.tensor([1, 0])
    A3_edges = torch.tensor([[0, 2],
                             [3, 0]])
    A4_nodes = torch.tensor([1, 0])
    A4_edges = torch.tensor([[0, 0],
                             [2, 0]])

    node_list = [A1_nodes, A2_nodes, A3_nodes, A4_nodes]
    node_features = [x[:, None].repeat(1, 4) for x in node_list]
    edge_list = [A1_edges, A2_edges, A3_edges, A4_edges]
    edge_features = [e.to_sparse().coalesce() for e in edge_list]
    edge_indices = [e.indices() for e in edge_features]
    edge_features = [e.values()[:, None].repeat(1, 4) for e in edge_features]
    G = make_batch(node_features, edge_indices, edge_features)

    G_nd = nd(G)

    A1_nodes = torch.tensor([0, 0, 0, 0])
    A1_edges = torch.tensor([[0, 2, 0, 3],
                             [4, 0, 6, 7],
                             [0, 8, 0, 8],
                             [7, 0, 6, 0]])
    A2_nodes = torch.tensor([0])
    A2_edges = torch.tensor([[0]])
    A3_nodes = torch.tensor([0, 0])
    A3_edges = torch.tensor([[0, 2],
                             [3, 0]])
    A4_nodes = torch.tensor([0, 0])
    A4_edges = torch.tensor([[0, 0],
                             [2, 0]])

    node_list = [A1_nodes, A2_nodes, A3_nodes, A4_nodes]
    edge_list = [A1_edges, A2_edges, A3_edges, A4_edges]

    for i, v, m, n, x, e in zip(G_nd.indices, G_nd.values, G_nd.mask, G_nd.n_nodes, node_list, edge_list):
        A_dense = torch.sparse_coo_tensor(i[m].t(), v[m][..., 4], size=(n, n)).to_dense()
        assert (A_dense == (x.diag() + e)).all()
