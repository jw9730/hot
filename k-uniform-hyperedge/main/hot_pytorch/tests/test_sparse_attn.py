import time

import torch

from ..batch.sparse import Batch, make_batch
from ..models.common import KernelFeatureMap

from ..models.sparse.attncoef import AttnCoef, apply_attn
from ..models.sparse.kernelattncoef import KernelFeatureMapWrapper, KernelAttnCoef

from ..models.sparse import SelfAttn, KernelSelfAttn, AvgPool
from ..models.encoder import EncLayer


def test_attn():
    G0 = torch.ones(4, 28)

    n_nodes = [2, 3, 4, 5]
    G1 = Batch(None, torch.ones(4, 5, 28), n_nodes, None, None)

    A1_nodes = torch.tensor([1, 1]).float()
    A1_edges = torch.tensor([[0, 0],
                             [0, 0]]).float()
    A2_nodes = torch.tensor([1, 1, 1]).float()
    A2_edges = torch.tensor([[0, 0, 1],
                             [1, 0, 1],
                             [0, 1, 0]]).float()
    A3_nodes = torch.tensor([1, 1, 1, 1]).float()
    A3_edges = torch.tensor([[0, 1, 1, 1],
                             [0, 0, 1, 1],
                             [1, 0, 0, 0],
                             [0, 1, 1, 0]]).float()
    A4_nodes = torch.tensor([1, 1, 1, 1, 1]).float()
    A4_edges = torch.tensor([[0, 1, 0, 1, 0],
                             [1, 0, 0, 1, 1],
                             [0, 1, 0, 1, 0],
                             [1, 0, 1, 0, 1],
                             [1, 0, 1, 1, 0]]).float()

    node_list = [A1_nodes, A2_nodes, A3_nodes, A4_nodes]
    node_features = [x[:, None].repeat(1, 14) for x in node_list]
    edge_list = [A1_edges, A2_edges, A3_edges, A4_edges]
    edge_features = [e.to_sparse().coalesce() for e in edge_list]
    edge_indices = [e.indices() for e in edge_features]
    edge_features = [e.values()[:, None].repeat(1, 14) for e in edge_features]
    G2 = make_batch(node_features, edge_indices, edge_features)
    v1, v2 = G2.values[..., :14].clone(), G2.values[..., 14:].clone()
    G2.values[..., :14] += v2
    G2.values[..., 14:] += v1

    alpha01 = AttnCoef(0, 1, 28, 7)(G0, G1)
    alpha02 = AttnCoef(0, 2, 28, 7)(G0, G2)
    alpha11, exp11 = AttnCoef(1, 1, 28, 7)(G1, G1, get_exp=True)
    alpha12 = AttnCoef(1, 2, 28, 7)(G1, G2)
    alpha21, exp21 = AttnCoef(2, 1, 28, 7)(G2, G1, get_exp=True)
    alpha22 = AttnCoef(2, 2, 28, 7)(G2, G2)
    assert alpha01.shape == torch.Size([7, 4, 5])
    assert alpha02.shape == torch.Size([7, 4, 18])
    assert alpha11.shape == torch.Size([7, 4, 5, 5]) == exp11.shape
    assert alpha12.shape == torch.Size([7, 4, 5, 18])
    assert alpha21.shape == torch.Size([7, 4, 18, 5]) == exp21.shape
    assert alpha22.shape == torch.Size([7, 4, 18, 18])
    print('alpha_0_1')
    print(alpha01[0, :, :])
    print('alpha_0_2')
    print(alpha02[0, :, :])
    print('alpha_1_1')
    print(alpha11[0, :, 0, :])
    print('alpha_1_2')
    print(alpha12[0, :, 0, :])
    print('alpha_2_1')
    print(alpha21[0, :, -1, :])
    print('alpha_2_2')
    print(alpha22[0, :, -1, :])

    A011 = apply_attn(G0, 1, alpha01, G1)
    A022 = apply_attn(G0, 2, alpha02, G2)
    A111 = apply_attn(G1, 1, alpha11, G1)
    A112_d12 = apply_attn(G1, 1, exp11, G2, (1, 2))
    A122 = apply_attn(G1, 2, alpha12, G2)
    A211 = apply_attn(G2, 1, alpha21, G1)
    A212_d13 = apply_attn(G2, 1, exp21, G2, (1, 3))
    A212_d23 = apply_attn(G2, 1, exp21, G2, (2, 3))
    A222 = apply_attn(G2, 2, alpha22, G2)
    assert A011.shape == torch.Size([4, 28])
    assert A022.shape == torch.Size([4, 28])
    assert A111.values.shape == torch.Size([4, 5, 28])
    assert A112_d12.values.shape == torch.Size([4, 5, 28])
    assert A122.values.shape == torch.Size([4, 5, 28])
    assert A211.values.shape == torch.Size([4, 18, 28])
    assert A212_d13.values.shape == torch.Size([4, 18, 28])
    assert A212_d23.values.shape == torch.Size([4, 18, 28])
    assert A222.values.shape == torch.Size([4, 18, 28])
    print('A011')
    print(A011[..., 0])
    print('A022')
    print(A022[..., 0])
    print('A111')
    print(A111.values[..., 0])
    print('A112_d12')
    print(A112_d12.values[..., 0])
    print('A122')
    print(A122.values[..., 0])
    print('A211')
    print(A211.values[..., 0])
    print('A212_d13')
    print(A212_d13.values[..., 0])
    print('A212_d23')
    print(A212_d23.values[..., 0])
    print('A222')
    print(A222.values[..., 0])


def test_kernel_attn():
    n_nodes = [2, 3, 4, 5]
    G1 = Batch(None, torch.ones(4, 5, 28), n_nodes, None, None)

    A1_nodes = torch.tensor([1, 1]).float()
    A1_edges = torch.tensor([[0, 0],
                             [0, 0]]).float()
    A2_nodes = torch.tensor([1, 1, 1]).float()
    A2_edges = torch.tensor([[0, 0, 1],
                             [1, 0, 1],
                             [0, 1, 0]]).float()
    A3_nodes = torch.tensor([1, 1, 1, 1]).float()
    A3_edges = torch.tensor([[0, 1, 1, 1],
                             [0, 0, 1, 1],
                             [1, 0, 0, 0],
                             [0, 1, 1, 0]]).float()
    A4_nodes = torch.tensor([1, 1, 1, 1, 1]).float()
    A4_edges = torch.tensor([[0, 1, 0, 1, 0],
                             [1, 0, 0, 1, 1],
                             [0, 1, 0, 1, 0],
                             [1, 0, 1, 0, 1],
                             [1, 0, 1, 1, 0]]).float()

    node_list = [A1_nodes, A2_nodes, A3_nodes, A4_nodes]
    node_features = [x[:, None].repeat(1, 14) for x in node_list]
    edge_list = [A1_edges, A2_edges, A3_edges, A4_edges]
    edge_features = [e.to_sparse().coalesce() for e in edge_list]
    edge_indices = [e.indices() for e in edge_features]
    edge_features = [e.values()[:, None].repeat(1, 14) for e in edge_features]
    G2 = make_batch(node_features, edge_indices, edge_features)
    v1, v2 = G2.values[..., :14].clone(), G2.values[..., 14:].clone()
    G2.values[..., :14] += v2
    G2.values[..., 14:] += v1

    feature_map = KernelFeatureMapWrapper(KernelFeatureMap(4), 28, 7)
    feat_dim = KernelFeatureMap(4).num_features
    q1 = feature_map(G1, is_query=True)
    q2 = feature_map(G2, is_query=True)
    k1 = feature_map(G1, is_query=False)
    k2 = feature_map(G2, is_query=False)

    att11 = KernelAttnCoef(1, 1, feat_dim, 28, 7)
    att12 = KernelAttnCoef(1, 2, feat_dim, 28, 7)
    att21 = KernelAttnCoef(2, 1, feat_dim, 28, 7)
    att22 = KernelAttnCoef(2, 2, feat_dim, 28, 7)

    alpha11 = att11.get_attn_coef(q1, k1)
    alpha12 = att12.get_attn_coef(q1, k2)
    alpha21 = att21.get_attn_coef(q2, k1)
    alpha22 = att22.get_attn_coef(q2, k2)
    assert alpha11.shape == torch.Size([7, 4, 5, 5])
    assert alpha12.shape == torch.Size([7, 4, 5, 18])
    assert alpha21.shape == torch.Size([7, 4, 18, 5])
    assert alpha22.shape == torch.Size([7, 4, 18, 18])
    print('alpha_1_1')
    print(alpha11[0, :, 0, :])
    print('alpha_1_2')
    print(alpha12[0, :, 0, :])
    print('alpha_2_1')
    print(alpha21[0, :, -1, :])
    print('alpha_2_2')
    print(alpha22[0, :, -1, :])

    A111 = att11(q1, k1, G1)
    A112_d12 = att11(q1, k1, G2, (1, 2))
    A122 = att12(q1, k2, G2)
    A211 = att21(q2, k1, G1)
    A212_d13 = att21(q2, k1, G2, (1, 3))
    A212_d23 = att21(q2, k1, G2, (2, 3))
    A222 = att22(q2, k2, G2)
    assert A111.values.shape == torch.Size([4, 5, 28])
    assert A112_d12.values.shape == torch.Size([4, 5, 28])
    assert A122.values.shape == torch.Size([4, 5, 28])
    assert A211.values.shape == torch.Size([4, 18, 28])
    assert A212_d13.values.shape == torch.Size([4, 18, 28])
    assert A212_d23.values.shape == torch.Size([4, 18, 28])
    assert A222.values.shape == torch.Size([4, 18, 28])
    print('A111')
    print(A111.values[..., 0])
    print('A112_d12')
    print(A112_d12.values[..., 0])
    print('A122')
    print(A122.values[..., 0])
    print('A211')
    print(A211.values[..., 0])
    print('A212_d13')
    print(A212_d13.values[..., 0])
    print('A212_d23')
    print(A212_d23.values[..., 0])
    print('A222')
    print(A222.values[..., 0])


def test_selfattn():
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
    a10 = SelfAttn(1, 0, 10, 20, 20, 4, 'default', 0., 0.).to(device)
    tic = time.time()
    assert a10(G1).shape == torch.Size([8, 10])
    print(time.time() - tic)

    print('1->1')
    a11 = SelfAttn(1, 1, 10, 20, 20, 4, 'default', 0., 0.).to(device)
    tic = time.time()
    assert a11(G1).values.shape == torch.Size([8, 100, 10])
    print(time.time() - tic)

    print('2->0')
    a20 = SelfAttn(2, 0, 10, 20, 20, 4, 'default', 0., 0.).to(device)
    tic = time.time()
    assert a20(G2).shape == torch.Size([8, 10])
    print(time.time() - tic)

    print('2->1')
    a21 = SelfAttn(2, 1, 10, 20, 20, 4, 'default', 0., 0.).to(device)
    tic = time.time()
    assert a21(G2).values.shape == torch.Size([8, 100, 10])
    print(time.time() - tic)

    print('2->2')
    a22 = SelfAttn(2, 2, 10, 20, 20, 4, 'default', 0., 0.).to(device)
    tic = time.time()
    assert a22(G2).values.shape == torch.Size([8, N_e, 10])
    print(time.time() - tic)


def test_kernelselfattn():
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

    feature_map = KernelFeatureMap(5)

    print('1->1')
    a11 = KernelSelfAttn(1, 1, 10, 20, 20, 4, 'default', 0., 0., feature_map).to(device)
    tic = time.time()
    assert a11(G1).values.shape == torch.Size([8, 100, 10])
    print(time.time() - tic)

    print('2->1')
    a21 = KernelSelfAttn(2, 1, 10, 20, 20, 4, 'default', 0., 0., feature_map).to(device)
    tic = time.time()
    assert a21(G2).values.shape == torch.Size([8, 100, 10])
    print(time.time() - tic)

    print('2->2')
    a22 = KernelSelfAttn(2, 2, 10, 20, 20, 4, 'default', 0., 0., feature_map).to(device)
    tic = time.time()
    assert a22(G2).values.shape == torch.Size([8, N_e, 10])
    print(time.time() - tic)


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

    device = 'cuda'
    G1.to(device)
    G2.to(device)
    p1 = AvgPool(1).to(device)
    p2 = AvgPool(2).to(device)

    def assert_grad(model):
        grads = []
        for param in model.parameters():
            grads.append(param.grad.view(-1))
        grads = torch.cat(grads)
        assert not grads.isnan().any(), 'gradient is nan'
        assert not grads.isinf().any(), 'gradient is inf'

    feature_map = KernelFeatureMap(5)

    print('1->0')
    l1 = EncLayer(1, 0, 10, 20, 20, 20, 4, 'default', 'default', 0., 0.).to(device)
    scalar = l1(G1).sum()
    tic = time.time()
    scalar.backward()
    print(time.time() - tic)
    assert_grad(l1)

    print('1->1')
    l1 = EncLayer(1, 1, 10, 20, 20, 20, 4, 'default', 'default', 0., 0.).to(device)
    l2 = EncLayer(1, 1, 10, 20, 20, 20, 4, 'default', 'kernel', 0., 0., feature_map).to(device)
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

    print('2->0')
    l1 = EncLayer(2, 0, 10, 20, 20, 20, 4, 'default', 'default', 0., 0., sparse=True).to(device)
    scalar = l1(G2).sum()
    tic = time.time()
    scalar.backward()
    print(time.time() - tic)
    assert_grad(l1)

    print('2->1')
    l1 = EncLayer(2, 1, 10, 20, 20, 20, 4, 'default', 'default', 0., 0., sparse=True).to(device)
    l2 = EncLayer(2, 1, 10, 20, 20, 20, 4, 'default', 'kernel', 0., 0., feature_map, sparse=True).to(device)
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

    print('2->2')
    l1 = EncLayer(2, 2, 10, 20, 20, 20, 4, 'default', 'kernel', 0., 0., feature_map, sparse=True).to(device)
    scalar = p2(l1(G2)).sum()
    tic = time.time()
    scalar.backward()
    print(time.time() - tic)
    assert_grad(l1)
