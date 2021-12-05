import time

import torch

from ..batch.dense import Batch
from ..models.common import KernelFeatureMap

from ..models.dense.attncoef import AttnCoef, apply_attn
from ..models.dense.kernelattncoef import KernelFeatureMapWrapper, KernelAttnCoef

from ..models.dense import SelfAttn, KernelSelfAttn, AvgPool
from ..models.encoder import EncLayer


def test_attn():
    n_nodes = [2, 3, 4, 5]
    G0 = torch.ones(4, 28)
    G1 = Batch(torch.ones(4, 5, 28), n_nodes, None)
    G2 = Batch(torch.ones(4, 5, 5, 28), n_nodes, None)

    alpha01 = AttnCoef(0, 1, 28, 7)(G0, G1)
    alpha02 = AttnCoef(0, 2, 28, 7)(G0, G2)
    alpha11 = AttnCoef(1, 1, 28, 7)(G1, G1)
    alpha12 = AttnCoef(1, 2, 28, 7)(G1, G2)
    alpha21 = AttnCoef(2, 1, 28, 7)(G2, G1)
    alpha22 = AttnCoef(2, 2, 28, 7)(G2, G2)
    assert alpha01.shape == torch.Size([7, 4, 5])
    assert alpha02.shape == torch.Size([7, 4, 25])
    assert alpha11.shape == torch.Size([7, 4, 5, 5])
    assert alpha12.shape == torch.Size([7, 4, 5, 25])
    assert alpha21.shape == torch.Size([7, 4, 25, 5])
    assert alpha22.shape == torch.Size([7, 4, 25, 25])
    print('alpha_0_1')
    print(alpha01[0, :, :])
    print('alpha_0_2')
    print(alpha02[0, :, :].view(4, 5, 5))
    print('alpha_1_1')
    print(alpha11[0, :, 1, :])
    print('alpha_1_2')
    print(alpha12[0, :, 1, :].view(4, 5, 5))
    print('alpha_2_1')
    print(alpha21[0, :, 1, :])
    print('alpha_2_2')
    print(alpha22[0, :, 1, :].view(4, 5, 5))

    A011 = apply_attn(0, 1, alpha01, G1)
    A022 = apply_attn(0, 2, alpha02, G2)
    A111 = apply_attn(1, 1, alpha11, G1)
    A112_d12 = apply_attn(1, 1, alpha11, G2, (1, 2))
    A122 = apply_attn(1, 2, alpha12, G2)
    A211 = apply_attn(2, 1, alpha21, G1)
    A212_d13 = apply_attn(2, 1, alpha21, G2, (1, 3))
    A212_d23 = apply_attn(2, 1, alpha21, G2, (2, 3))
    A222 = apply_attn(2, 2, alpha22, G2)
    assert A011.shape == torch.Size([4, 28])
    assert A022.shape == torch.Size([4, 28])
    assert A111.A.shape == torch.Size([4, 5, 28])
    assert A112_d12.A.shape == torch.Size([4, 5, 28])
    assert A122.A.shape == torch.Size([4, 5, 28])
    assert A211.A.shape == torch.Size([4, 5, 5, 28])
    assert A212_d13.A.shape == torch.Size([4, 5, 5, 28])
    assert A212_d23.A.shape == torch.Size([4, 5, 5, 28])
    assert A222.A.shape == torch.Size([4, 5, 5, 28])
    print('A011')
    print(A011[..., 0])
    print('A022')
    print(A022[..., 0])
    print('A111')
    print(A111.A[..., 0])
    print('A112_d12')
    print(A112_d12.A[..., 0])
    print('A122')
    print(A122.A[..., 0])
    print('A211')
    print(A211.A[..., 0])
    print('A212_d13')
    print(A212_d13.A[..., 0])
    print('A212_d23')
    print(A212_d23.A[..., 0])
    print('A222')
    print(A222.A[..., 0])


def test_kernel_attn():
    n_nodes = [2, 3, 4, 5]
    G1 = Batch(torch.ones(4, 5, 28), n_nodes, None)
    G2 = Batch(torch.ones(4, 5, 5, 28), n_nodes, None)

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
    assert alpha12.shape == torch.Size([7, 4, 5, 25])
    assert alpha21.shape == torch.Size([7, 4, 25, 5])
    assert alpha22.shape == torch.Size([7, 4, 25, 25])
    print('alpha_1_1')
    print(alpha11[0, :, 1, :])
    print('alpha_1_2')
    print(alpha12[0, :, 1, :].view(4, 5, 5))
    print('alpha_2_1')
    print(alpha21[0, :, 1, :])
    print('alpha_2_2')
    print(alpha22[0, :, 1, :].view(4, 5, 5))

    A111 = att11(q1, k1, G1)
    A112_d12 = att11(q1, k1, G2, (1, 2))
    A122 = att12(q1, k2, G2)
    A211 = att21(q2, k1, G1)
    A212_d13 = att21(q2, k1, G2, (1, 3))
    A212_d23 = att21(q2, k1, G2, (2, 3))
    A222 = att22(q2, k2, G2)
    assert A111.A.shape == torch.Size([4, 5, 28])
    assert A112_d12.A.shape == torch.Size([4, 5, 28])
    assert A122.A.shape == torch.Size([4, 5, 28])
    assert A211.A.shape == torch.Size([4, 5, 5, 28])
    assert A212_d13.A.shape == torch.Size([4, 5, 5, 28])
    assert A212_d23.A.shape == torch.Size([4, 5, 5, 28])
    assert A222.A.shape == torch.Size([4, 5, 5, 28])

    print('A111')
    print(A111.A[..., 0])
    print('A112_d12')
    print(A112_d12.A[..., 0])
    print('A122')
    print(A122.A[..., 0])
    print('A211')
    print(A211.A[..., 0])
    print('A212_d13')
    print(A212_d13.A[..., 0])
    print('A212_d23')
    print(A212_d23.A[..., 0])
    print('A222')
    print(A222.A[..., 0])


def test_selfattn():
    n_nodes = [40, 60, 80, 100, 40, 60, 80, 100]
    G1 = Batch(torch.ones(8, 100, 10), n_nodes, None)
    G2 = Batch(torch.ones(8, 100, 100, 10), n_nodes, None)
    n_nodes = [4, 6, 8, 10, 4, 6, 8, 10]
    G1 = Batch(torch.ones(8, 10, 10), n_nodes, None)
    G2 = Batch(torch.ones(8, 10, 10, 10), n_nodes, None)

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
    # assert a11(G1).A.shape == torch.Size([8, 100, 10])
    assert a11(G1).A.shape == torch.Size([8, 10, 10])
    print(time.time() - tic)

    print('1->2')
    a12 = SelfAttn(1, 2, 10, 20, 20, 4, 'default', 0., 0.).to(device)
    tic = time.time()
    # assert a12(G1).A.shape == torch.Size([8, 100, 100, 10])
    assert a12(G1).A.shape == torch.Size([8, 10, 10, 10])
    print(time.time() - tic)

    print('2->0')
    a20 = SelfAttn(2, 0, 10, 20, 20, 4, 'default', 0., 0.).to(device)
    tic = time.time()
    assert a20(G2).shape == torch.Size([8, 10])
    print(time.time() - tic)

    print('2->1')
    a21 = SelfAttn(2, 1, 10, 20, 20, 4, 'default', 0., 0.).to(device)
    tic = time.time()
    # assert a21(G2).A.shape == torch.Size([8, 100, 10])
    assert a21(G2).A.shape == torch.Size([8, 10, 10])
    print(time.time() - tic)

    print('2->2')
    a22 = SelfAttn(2, 2, 10, 20, 20, 4, 'default', 0., 0.).to(device)
    tic = time.time()
    # assert a22(G2).A.shape == torch.Size([8, 100, 100, 10])
    assert a22(G2).A.shape == torch.Size([8, 10, 10, 10])
    print(time.time() - tic)


def test_kernelselfattn():
    n_nodes = [40, 60, 80, 100, 40, 60, 80, 100]
    G1 = Batch(torch.ones(8, 100, 10), n_nodes, None)
    G2 = Batch(torch.ones(8, 100, 100, 10), n_nodes, None)
    n_nodes = [4, 6, 8, 10, 4, 6, 8, 10]
    G1 = Batch(torch.ones(8, 10, 10), n_nodes, None)
    G2 = Batch(torch.ones(8, 10, 10, 10), n_nodes, None)

    device = 'cuda'
    G1.to(device)
    G2.to(device)

    feature_map = KernelFeatureMap(5)

    print('1->1')
    a11 = KernelSelfAttn(1, 1, 10, 20, 20, 4, 'default', 0., 0., feature_map).to(device)
    tic = time.time()
    # assert a11(G1).A.shape == torch.Size([8, 100, 10])
    assert a11(G1).A.shape == torch.Size([8, 10, 10])
    print(time.time() - tic)

    print('1->2')
    a12 = KernelSelfAttn(1, 2, 10, 20, 20, 4, 'default', 0., 0., feature_map).to(device)
    tic = time.time()
    # assert a12(G1).A.shape == torch.Size([8, 100, 100, 10])
    assert a12(G1).A.shape == torch.Size([8, 10, 10, 10])
    print(time.time() - tic)

    print('2->1')
    a21 = KernelSelfAttn(2, 1, 10, 20, 20, 4, 'default', 0., 0., feature_map).to(device)
    tic = time.time()
    # assert a21(G2).A.shape == torch.Size([8, 100, 10])
    assert a21(G2).A.shape == torch.Size([8, 10, 10])
    print(time.time() - tic)

    print('2->2')
    a22 = KernelSelfAttn(2, 2, 10, 20, 20, 4, 'default', 0., 0., feature_map).to(device)
    tic = time.time()
    # assert a22(G2).A.shape == torch.Size([8, 100, 100, 10])
    assert a22(G2).A.shape == torch.Size([8, 10, 10, 10])
    print(time.time() - tic)


def test_backward():
    n_nodes = [40, 60, 80, 100, 40, 60, 80, 100]
    G1 = Batch(torch.ones(8, 100, 10), n_nodes, None)
    G2 = Batch(torch.ones(8, 100, 100, 10), n_nodes, None)
    n_nodes = [4, 6, 8, 10, 4, 6, 8, 10]
    G1 = Batch(torch.ones(8, 10, 10), n_nodes, None)
    G2 = Batch(torch.ones(8, 10, 10, 10), n_nodes, None)

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
    l1 = EncLayer(1, 0, 10, 20, 20, 20, 4, 'default', 'default', 0., 0., sparse=False).to(device)
    scalar = l1(G1).sum()
    tic = time.time()
    scalar.backward()
    print(time.time() - tic)
    assert_grad(l1)

    print('1->1')
    l1 = EncLayer(1, 1, 10, 20, 20, 20, 4, 'default', 'default', 0., 0., sparse=False).to(device)
    l2 = EncLayer(1, 1, 10, 20, 20, 20, 4, 'default', 'kernel', 0., 0., feature_map, sparse=False).to(device)
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

    print('1->2')
    l1 = EncLayer(1, 2, 10, 20, 20, 20, 4, 'default', 'default', 0., 0., sparse=False).to(device)
    l2 = EncLayer(1, 2, 10, 20, 20, 20, 4, 'default', 'kernel', 0., 0., feature_map, sparse=False).to(device)
    scalar = p2(l1(G1)).sum()
    tic = time.time()
    scalar.backward()
    print(time.time() - tic)
    assert_grad(l1)
    scalar = p2(l2(G1)).sum()
    tic = time.time()
    scalar.backward()
    print(time.time() - tic)
    assert_grad(l2)

    print('2->0')
    l1 = EncLayer(2, 0, 10, 20, 20, 20, 4, 'default', 'default', 0., 0., sparse=False).to(device)
    scalar = l1(G2).sum()
    tic = time.time()
    scalar.backward()
    print(time.time() - tic)
    assert_grad(l1)

    print('2->1')
    l1 = EncLayer(2, 1, 10, 20, 20, 20, 4, 'default', 'default', 0., 0., sparse=False).to(device)
    l2 = EncLayer(2, 1, 10, 20, 20, 20, 4, 'default', 'kernel', 0., 0., feature_map, sparse=False).to(device)
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
    l1 = EncLayer(2, 2, 10, 20, 20, 20, 4, 'default', 'kernel', 0., 0., feature_map, sparse=False).to(device)
    scalar = p2(l1(G2)).sum()
    tic = time.time()
    scalar.backward()
    print(time.time() - tic)
    assert_grad(l1)
