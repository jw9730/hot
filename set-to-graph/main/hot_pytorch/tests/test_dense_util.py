import torch

from ..utils.set import to_masked_batch
from ..utils.dense import to_diag, get_diag, get_nondiag, rotate
from ..batch.dense import Batch, t, v2d, nd, d, add_batch


def test_diag():
    x = torch.randn(12, 20)
    n_nodes = [1, 2, 3, 4, 2]
    masked_x, mask = to_masked_batch(x, n_nodes)

    expanded_x = to_diag(masked_x)
    recovered_x = get_diag(expanded_x)

    assert expanded_x.size() == torch.Size([5, 4, 4, 20])
    assert recovered_x.size() == torch.Size([5, 4, 20])
    assert (masked_x == recovered_x).all()


def test_nondiag():
    x = torch.randn(4, 8, 8, 10)
    assert (x == to_diag(get_diag(x)) + get_nondiag(x)).all()


def test_rotate():
    x = torch.randn(4, 1, 3, 6, 5)

    assert rotate(x, 4, 1).size() == torch.Size([4, 5, 1, 3, 6])
    assert rotate(x, 1, 4).size() == torch.Size([4, 3, 6, 5, 1])
    assert (x == rotate(rotate(x, 3, 0), 0, 3)).all()

    x = torch.randn(4, 4)
    assert (x == rotate(x, 0, 0)).all()
    assert (x.t() == rotate(x, 1, 0)).all()


def test_batch():
    x = torch.randn(12, 20)
    n_nodes = [1, 2, 3, 4, 2]
    masked_x, mask = to_masked_batch(x, n_nodes)

    G1 = Batch(masked_x, n_nodes, mask)
    G2 = Batch(masked_x, n_nodes, None)
    assert G1.order == G2.order == 1
    assert (mask == G1.mask).all()
    assert (G1.mask == G2.mask).all()
    for m_x, x, m in zip(masked_x, G1.A, mask):
        assert (m_x[m] == x[m]).all()

    expanded_x = to_diag(masked_x)
    G3 = Batch(expanded_x, n_nodes, None)
    assert G3.order == 2
    assert G3.mask.size() == torch.Size([5, 4, 4])


def test_batch_fn():
    x = torch.randn(5, 4, 4, 10)
    n_nodes = [1, 2, 3, 4, 2]
    G = Batch(x, n_nodes, None)

    assert (G.A.transpose(2, 1) == t(G).A).all()
    assert (get_nondiag(G.A) == nd(G).A).all()
    assert (to_diag(get_diag(G.A)) == v2d(d(G)).A).all()
    assert (to_diag(get_diag(G.A)) == v2d(d(G), G.mask).A).all()  # faster
    assert (G.A == add_batch(nd(G), v2d(d(G))).A).all()
