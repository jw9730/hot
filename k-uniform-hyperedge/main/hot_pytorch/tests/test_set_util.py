import torch

from ..utils.set import to_masked_batch, test_valid_mask, masked_fill, MASK


def test_masking():
    x = torch.randn(10, 20)
    n_nodes = [1, 2, 3, 4]

    masked_x, mask = to_masked_batch(x, n_nodes)

    assert masked_x.size() == torch.Size([4, 4, 20])
    assert mask.size() == torch.Size([4, 4])

    test_valid_mask(masked_x, mask)

    masked_fill(masked_x, mask, MASK)
    test_valid_mask(masked_x, mask)

    assert (masked_x[0, :1] == x[:1]).all()
    assert (masked_x[1, :2] == x[1:3]).all()
    assert (masked_x[2, :3] == x[3:6]).all()
    assert (masked_x[3, :4] == x[6:]).all()
