import time

import torch

from ..utils.set import get_mask
from ..utils.dense import to_diag, get_nondiag, get_diag, rotate
from ..batch.dense import Batch
from ..models.dense.masksum import mask_tensor, do_masked_sum
from ..models.dense import Linear, SumPool, AvgPool, MaxPool


def test_mask():
    n = 10
    M2 = mask_tensor(2, n)
    M3 = mask_tensor(3, n)
    M4 = mask_tensor(4, n)

    assert (M2 == rotate(M2, src_dim=1, tgt_dim=0)).all()

    assert (M3 == rotate(M3, src_dim=1, tgt_dim=0)).all()
    assert (M3 == rotate(M3, src_dim=2, tgt_dim=1)).all()
    assert (M3 == rotate(M3, src_dim=0, tgt_dim=1)).all()
    assert (M3 == rotate(M3, src_dim=1, tgt_dim=2)).all()

    assert (M4 == rotate(M4, src_dim=0, tgt_dim=1)).all()
    assert (M4 == rotate(M4, src_dim=0, tgt_dim=3)).all()
    assert (M4 == rotate(M4, src_dim=1, tgt_dim=3)).all()
    assert (M4 == rotate(M4, src_dim=1, tgt_dim=0)).all()
    assert (M4 == rotate(M4, src_dim=3, tgt_dim=1)).all()
    assert (M4 == rotate(M4, src_dim=3, tgt_dim=1)).all()


def test_1_0():
    x = torch.ones(5, 4, 10)
    n_nodes = [1, 2, 3, 4, 2]
    G = Batch(x, n_nodes, None)

    A1 = G.A.sum(1)  # [B, N, D] -> [B, D]
    assert ((torch.tensor(n_nodes) - A1[..., 0]).abs() <= 1e-4).all()

    n_vec = torch.tensor(G.n_nodes).to(A1).unsqueeze(-1)  # [B, 1]
    n_vec = n_vec.masked_fill_(n_vec == 0, 1e-5)
    A1 = A1 / n_vec  # [B, D]
    assert ((1 - A1[..., 0]).abs() <= 1e-4).all()


def test_1_1():
    x = torch.ones(5, 4, 10)
    n_nodes = [1, 2, 3, 4, 2]
    G = Batch(x, n_nodes, None)

    M_2 = mask_tensor(2, 4)  # [N, N]

    A = do_masked_sum(M_2, G.A, G.mask, normalize=False)  # [B, N, D]
    G_sum = Batch(A, n_nodes, None)
    A_ = torch.tensor([[0, 0, 0, 0],
                       [1, 1, 0, 0],
                       [2, 2, 2, 0],
                       [3, 3, 3, 3],
                       [1, 1, 0, 0]])
    assert ((G_sum.A[..., 0] - A_).abs() < 1e-4).all()

    A = do_masked_sum(M_2, G.A, G.mask, normalize=True)  # [B, N, D]
    G_mean = Batch(A, n_nodes, None)
    A_ = torch.tensor([[0, 0, 0, 0],
                       [1, 1, 0, 0],
                       [1, 1, 1, 0],
                       [1, 1, 1, 1],
                       [1, 1, 0, 0]])
    assert ((G_mean.A[..., 0] - A_).abs() < 1e-4).all()


def test_1_2():
    x = torch.ones(5, 4, 10)
    n_nodes = [1, 2, 3, 4, 2]
    G = Batch(x, n_nodes, None)
    A_ = torch.tensor([1, 1, 1])
    assert ((G.A[2, 0:3, 0] - A_).abs() < 1e-4).all()
    G.A[2, 0:3, 0] = torch.tensor([1, 2, 3])

    M_2 = mask_tensor(2, 4)
    M_3 = mask_tensor(3, 4)

    A1 = to_diag(G.A)
    A1_ = torch.tensor([[1, 0, 0],
                        [0, 2, 0],
                        [0, 0, 3]])
    assert ((A1[2, 0:3, 0:3, 0] - A1_).abs() < 1e-4).all()

    A2 = get_nondiag(G.A.unsqueeze(1).repeat(1, 4, 1, 1))
    A2_ = torch.tensor([[0, 2, 3],
                        [1, 0, 3],
                        [1, 2, 0]])
    assert ((A2[2, 0:3, 0:3, 0] - A2_).abs() < 1e-4).all()

    A3 = get_nondiag(G.A.unsqueeze(2).repeat(1, 1, 4, 1))
    A3_ = torch.tensor([[0, 1, 1],
                        [2, 0, 2],
                        [3, 3, 0]])
    assert ((A3[2, 0:3, 0:3, 0] - A3_).abs() < 1e-4).all()

    A4 = to_diag(do_masked_sum(M_2, G.A, G.mask, normalize=False))
    A4_ = torch.tensor([[5, 0, 0],
                        [0, 4, 0],
                        [0, 0, 3]])
    assert ((A4[2, 0:3, 0:3, 0] - A4_).abs() < 1e-4).all()

    A4 = to_diag(do_masked_sum(M_2, G.A, G.mask, normalize=True))
    A4_ = torch.tensor([[5/2, 0,   0],
                        [0,   2,   0],
                        [0,   0, 3/2]])
    assert ((A4[2, 0:3, 0:3, 0] - A4_).abs() < 1e-4).all()

    A5 = do_masked_sum(M_3, G.A, G.mask, normalize=False)
    A5_ = torch.tensor([[0, 3, 2],
                        [3, 0, 1],
                        [2, 1, 0]])
    assert ((A5[2, 0:3, 0:3, 0] - A5_).abs() < 1e-4).all()

    A5 = do_masked_sum(M_3, G.A, G.mask, normalize=True)
    A5_ = torch.tensor([[0, 3, 2],
                        [3, 0, 1],
                        [2, 1, 0]])
    assert ((A5[2, 0:3, 0:3, 0] - A5_).abs() < 1e-4).all()


def test_2_0():
    x = torch.ones(5, 4, 4, 10)
    n_nodes = [1, 2, 3, 4, 2]
    G = Batch(x, n_nodes, None)

    A1 = get_diag(G.A).sum(1)  # [B, N, D] -> [B, D]
    A2 = get_nondiag(G.A).sum(1).sum(1)  # [B, N, N, D] -> [B, D]

    assert ((A1[:, 0] - torch.tensor([1, 2, 3, 4, 2])).abs() < 1e-4).all()
    assert ((A2[:, 0] - torch.tensor([0, 2, 6, 12, 2])).abs() < 1e-4).all()

    n_vec = torch.tensor(G.n_nodes, dtype=torch.float).to(G.A.device).unsqueeze(-1)  # [B, 1]
    A1 = A1 / n_vec.clone().masked_fill_(n_vec == 0, 1e-5)  # [B, D]
    e_vec = (n_vec.pow(2) - n_vec)  # [B, 1]
    A2 = A2 / e_vec.masked_fill_(e_vec == 0, 1e-5)  # [B, D]

    assert ((A1[:, 0] - torch.tensor([1, 1, 1, 1, 1])).abs() < 1e-4).all(), A1[:, 0]
    assert ((A2[:, 0] - torch.tensor([0, 1, 1, 1, 1])).abs() < 1e-4).all(), A2[:, 0]


def test_2_1():
    x = torch.ones(5, 4, 4, 10)
    n_nodes = [1, 2, 3, 4, 2]
    G = Batch(x, n_nodes, None)
    G.A[2, 0:3, 0:3, 0] = torch.tensor([[1, 2, 3],
                                        [1, 2, 3],
                                        [1, 2, 3]])
    diagonal = get_diag(G.A)

    node_mask = get_mask(torch.tensor(G.n_nodes, dtype=torch.long))
    M_2 = mask_tensor(2, 4)
    M_3 = mask_tensor(3, 4)

    A1 = diagonal
    assert ((A1[2, 0:3, 0] - torch.tensor([1, 2, 3])).abs() < 1e-4).all()

    # graph -> set
    A_AT = torch.cat([G.A, G.A.transpose(2, 1)], dim=-1)  # [B, N, N, 2D]
    A3_2 = do_masked_sum(M_2, A_AT, node_mask, l=1, normalize=False, diagonal=(1, 2))
    A3, A2 = A3_2[..., :10], A3_2[..., 10:]
    assert ((A2[2, 0:3, 0] - torch.tensor([5, 4, 3])).abs() < 1e-4).all()
    assert ((A3[2, 0:3, 0] - torch.tensor([2, 4, 6])).abs() < 1e-4).all()

    A4 = do_masked_sum(M_2, diagonal, node_mask, normalize=False)
    assert ((A4[2, 0:3, 0] - torch.tensor([5, 4, 3])).abs() < 1e-4).all()

    A5 = do_masked_sum(M_3, G.A, node_mask, normalize=False)
    assert ((A5[2, 0:3, 0] - torch.tensor([5, 4, 3])).abs() < 1e-4).all()

    A_AT = torch.cat([G.A, G.A.transpose(2, 1)], dim=-1)  # [B, N, N, 2D]
    A3_2 = do_masked_sum(M_2, A_AT, node_mask, l=1, normalize=True, diagonal=(1, 2))
    A3, A2 = A3_2[..., :10], A3_2[..., 10:]
    assert ((A2[2, 0:3, 0] - torch.tensor([5 / 2, 4 / 2, 3 / 2])).abs() < 1e-4).all()
    assert ((A3[2, 0:3, 0] - torch.tensor([1, 2, 3])).abs() < 1e-4).all()

    A4 = do_masked_sum(M_2, diagonal, node_mask, normalize=True)
    assert ((A4[2, 0:3, 0] - torch.tensor([5 / 2, 4 / 2, 3 / 2])).abs() < 1e-4).all()

    A5 = do_masked_sum(M_3, G.A, node_mask, normalize=True)
    assert ((A5[2, 0:3, 0] - torch.tensor([5 / 2, 4 / 2, 3 / 2])).abs() < 1e-4).all()


def test_2_2():
    x = torch.ones(5, 4, 4, 10)
    n_nodes = [1, 2, 3, 4, 2]
    G = Batch(x, n_nodes, None)
    G.A[3, :, :, 0] = torch.tensor([[1, 2, 3, 4],
                                    [3, 2, 1, 2],
                                    [3, 4, 3, 2],
                                    [1, 2, 3, 2]])
    diagonal = get_diag(G.A)

    node_mask = get_mask(torch.tensor(G.n_nodes, dtype=torch.long))
    M_3 = mask_tensor(3, 4)
    M_4 = mask_tensor(4, 4)

    A_AT = torch.cat([G.A, G.A.transpose(2, 1)], dim=-1)  # [B, N, 2D]
    A1_3 = do_masked_sum(M_3, A_AT, node_mask, l=1, normalize=False, diagonal=(2, 3))
    A1, A3 = A1_3[..., :10], A1_3[..., 10:]
    A4_2 = do_masked_sum(M_3, A_AT, node_mask, l=1, normalize=False, diagonal=(1, 3))
    A4, A2 = A4_2[..., :10], A4_2[..., 10:]
    # Oij (i != j) = sum_k!=i,j Akj
    assert ((A1[3, :, :, 0] - torch.tensor([[0, 6, 4, 4],
                                            [4, 0, 6, 6],
                                            [4, 4, 0, 6],
                                            [6, 6, 4, 0]])).abs() < 1e-4).all()
    # Oij (i != j) = sum_k!=i,j Aik
    assert ((A2[3, :, :, 0] - torch.tensor([[0, 7, 6, 5],
                                            [3, 0, 5, 4],
                                            [6, 5, 0, 7],
                                            [5, 4, 3, 0]])).abs() < 1e-4).all()
    # Oij (i != j) = sum_k!=i,j Ajk
    assert ((A3[3, :, :, 0] - torch.tensor([[0, 3, 6, 5],
                                            [7, 0, 5, 4],
                                            [6, 5, 0, 3],
                                            [5, 4, 7, 0]])).abs() < 1e-4).all()
    # Oij (i != j) = sum_k!=i,j Aki
    assert ((A4[3, :, :, 0] - torch.tensor([[0, 4, 4, 6],
                                            [6, 0, 4, 6],
                                            [4, 6, 0, 4],
                                            [4, 6, 6, 0]])).abs() < 1e-4).all()

    A5 = do_masked_sum(M_3, diagonal, node_mask, normalize=False)
    assert ((A5[3, :, :, 0] - torch.tensor([[0, 5, 4, 5],
                                            [5, 0, 3, 4],
                                            [4, 3, 0, 3],
                                            [5, 4, 3, 0]])).abs() < 1e-4).all()

    A6 = do_masked_sum(M_4, G.A, node_mask, normalize=False)
    assert ((A6[3, :, :, 0] - torch.tensor([[0, 5, 4, 5],
                                            [5, 0, 5, 6],
                                            [4, 5, 0, 5],
                                            [5, 6, 5, 0]])).abs() < 1e-4).all()

    G.A[3, :, :, 0] = torch.tensor([[1, 1, 1, 1],
                                    [1, 1, 1, 1],
                                    [1, 1, 1, 1],
                                    [1, 1, 1, 1]])
    diagonal = get_diag(G.A)

    A_AT = torch.cat([G.A, G.A.transpose(2, 1)], dim=-1)  # [B, N, 2D]
    A1_3 = do_masked_sum(M_3, A_AT, node_mask, l=1, normalize=True, diagonal=(2, 3))
    A1, A3 = A1_3[..., :10], A1_3[..., 10:]
    A4_2 = do_masked_sum(M_3, A_AT, node_mask, l=1, normalize=True, diagonal=(1, 3))
    A4, A2 = A4_2[..., :10], A4_2[..., 10:]
    # Oij (i != j) = sum_k!=i,j Akj
    assert ((A1[3, :, :, 0] - torch.tensor([[0, 1, 1, 1],
                                            [1, 0, 1, 1],
                                            [1, 1, 0, 1],
                                            [1, 1, 1, 0]])).abs() < 1e-4).all()
    # Oij (i != j) = sum_k!=i,j Aik
    assert ((A2[3, :, :, 0] - torch.tensor([[0, 1, 1, 1],
                                            [1, 0, 1, 1],
                                            [1, 1, 0, 1],
                                            [1, 1, 1, 0]])).abs() < 1e-4).all()
    # Oij (i != j) = sum_k!=i,j Ajk
    assert ((A3[3, :, :, 0] - torch.tensor([[0, 1, 1, 1],
                                            [1, 0, 1, 1],
                                            [1, 1, 0, 1],
                                            [1, 1, 1, 0]])).abs() < 1e-4).all()
    # Oij (i != j) = sum_k!=i,j Aki
    assert ((A4[3, :, :, 0] - torch.tensor([[0, 1, 1, 1],
                                            [1, 0, 1, 1],
                                            [1, 1, 0, 1],
                                            [1, 1, 1, 0]])).abs() < 1e-4).all()

    A5 = do_masked_sum(M_3, diagonal, node_mask, normalize=True)
    assert ((A5[3, :, :, 0] - torch.tensor([[0, 1, 1, 1],
                                            [1, 0, 1, 1],
                                            [1, 1, 0, 1],
                                            [1, 1, 1, 0]])).abs() < 1e-4).all()

    A6 = do_masked_sum(M_4, G.A, node_mask, normalize=True)
    assert ((A6[3, :, :, 0] - torch.tensor([[0, 1, 1, 1],
                                            [1, 0, 1, 1],
                                            [1, 1, 0, 1],
                                            [1, 1, 1, 0]])).abs() < 1e-4).all()


def test_forward():
    n_nodes = [40, 60, 80, 100, 40, 60, 80, 100]
    G1 = Batch(torch.ones(8, 100, 10), n_nodes, None)
    G2 = Batch(torch.ones(8, 100, 100, 10), n_nodes, None)

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
    assert l1(G1).A.shape == torch.Size([8, 100, 20])
    print(time.time() - tic)
    tic = time.time()
    assert l2(G1).A.shape == torch.Size([8, 100, 20])
    print(time.time() - tic)
    tic = time.time()
    assert l3(G1).A.shape == torch.Size([8, 100, 20])
    print(time.time() - tic)

    print('1->2')
    l1 = Linear(1, 2, 10, 20, True, 'default', False).to(device)
    l2 = Linear(1, 2, 10, 20, True, 'default', True).to(device)
    l3 = Linear(1, 2, 10, 20, True, 'light', False).to(device)
    tic = time.time()
    assert l1(G1).A.shape == torch.Size([8, 100, 100, 20])
    print(time.time() - tic)
    tic = time.time()
    assert l2(G1).A.shape == torch.Size([8, 100, 100, 20])
    print(time.time() - tic)
    tic = time.time()
    assert l3(G1).A.shape == torch.Size([8, 100, 100, 20])
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
    assert l1(G2).A.shape == torch.Size([8, 100, 20])
    print(time.time() - tic)
    tic = time.time()
    assert l2(G2).A.shape == torch.Size([8, 100, 20])
    print(time.time() - tic)
    tic = time.time()
    assert l3(G2).A.shape == torch.Size([8, 100, 20])
    print(time.time() - tic)

    print('2->2')
    l1 = Linear(2, 2, 10, 20, True, 'default', False).to(device)
    l2 = Linear(2, 2, 10, 20, True, 'default', True).to(device)
    l3 = Linear(2, 2, 10, 20, True, 'light', False).to(device)
    tic = time.time()
    assert l1(G2).A.shape == torch.Size([8, 100, 100, 20])
    print(time.time() - tic)
    tic = time.time()
    assert l2(G2).A.shape == torch.Size([8, 100, 100, 20])
    print(time.time() - tic)
    tic = time.time()
    assert l3(G2).A.shape == torch.Size([8, 100, 100, 20])
    print(time.time() - tic)


def test_pool():
    x = torch.ones(5, 4, 4, 10)
    n_nodes = [1, 2, 3, 4, 2]
    G = Batch(x, n_nodes, None)

    G.A[3, :, :, 0] = -torch.tensor([[4, 2, 3, 4],
                                     [3, 7, 1, 5],
                                     [3, 4, 5, 2],
                                     [1, 2, 3, 2]])
    assert (MaxPool(2)(G)[3, 0] + 3).abs() < 1e-4
    G.A[3, :, :, 0] = torch.tensor([[1, 1, 1, 1],
                                    [1, 1, 1, 1],
                                    [1, 1, 1, 1],
                                    [1, 1, 1, 1]])
    assert (SumPool(2)(G)[3, 0] - 16).abs() < 1e-4
    G.A[3, :, :, 0] = torch.tensor([[2, 1, 1, 1],
                                    [1, 2, 1, 1],
                                    [1, 1, 2, 1],
                                    [1, 1, 1, 2]])
    assert (AvgPool(2)(G)[3, 0] - 3).abs() < 1e-4

    G.A[2, :3, :3, 0] = -torch.tensor([[4, 2, 3],
                                       [3, 7, 1],
                                       [3, 4, 5]])
    assert (MaxPool(2)(G)[2, 0] + 5).abs() < 1e-4
    G.A[2, :3, :3, 0] = torch.tensor([[1, 1, 1],
                                      [1, 1, 1],
                                      [1, 1, 1]])
    assert (SumPool(2)(G)[2, 0] - 9).abs() < 1e-4
    G.A[2, :3, :3, 0] = torch.tensor([[2, 1, 1],
                                      [1, 2, 1],
                                      [1, 1, 2]])
    assert (AvgPool(2)(G)[2, 0] - 3).abs() < 1e-4

    x = torch.ones(5, 4, 10)
    n_nodes = [1, 2, 3, 4, 2]
    G = Batch(x, n_nodes, None)

    G.A[3, :, 0] = torch.tensor([[1, 2, 3, 4]])
    assert (MaxPool(1)(G)[3, 0] - 4).abs() < 1e-4
    assert (SumPool(1)(G)[3, 0] - 10).abs() < 1e-4
    assert (AvgPool(1)(G)[3, 0] - 2.5).abs() < 1e-4

    G.A[2, 0:3, 0] = torch.tensor([[-1, -2, -3]])
    assert (MaxPool(1)(G)[2, 0] + 1).abs() < 1e-4
    assert (SumPool(1)(G)[2, 0] + 6).abs() < 1e-4
    assert (AvgPool(1)(G)[2, 0] + 2).abs() < 1e-4


def test_backward():
    n_nodes = [40, 60, 80, 100, 40, 60, 80, 100]
    G1 = Batch(torch.ones(8, 100, 10), n_nodes, None)
    G2 = Batch(torch.ones(8, 100, 100, 10), n_nodes, None)

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

    print('1->2')
    l1 = Linear(1, 2, 10, 20, True, 'default', False).to(device)
    l2 = Linear(1, 2, 10, 20, True, 'default', True).to(device)
    l3 = Linear(1, 2, 10, 20, True, 'light', False).to(device)
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
    scalar = p2(l3(G1)).sum()
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
