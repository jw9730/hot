import torch

from ...utils.dense import rotate


@torch.no_grad()
def mask_tensor(order: int, n: int, cast_float: bool = True, device=None) -> torch.Tensor:
    """
    :param order: int
    :param n: int
    :param cast_float: bool
    :param device
    :return: Tensor([n^order])
    """
    assert order >= 2
    if order == 2:
        M = torch.ones(n, n, device=device) - torch.eye(n, device=device)
        return M if cast_float else M.bool()
    index_tensors = list()
    for idx in range(order):
        sizes = [1] * order
        sizes[idx] = n  # sizes: [1...n...1]
        index_tensors.append(torch.arange(n, device=device).reshape(sizes))
    # assign value 1 to a multi-index only if every elements are unique
    M = torch.ones([n] * order, device=device).bool()
    for axis1 in range(order):
        for axis2 in range(axis1):
            M = M & (index_tensors[axis1] != index_tensors[axis2])
    return M.float() if cast_float else M


def do_masked_sum(M: torch.Tensor, A: torch.Tensor, node_mask: torch.BoolTensor = None,
                  l: int = None, normalize: bool = False, diagonal: tuple = None) -> torch.Tensor:
    """
    :param M: Tensor([N^(k+l)])
    :param A: Tensor([B, N^(l+t), D])
    :param node_mask: BoolTensor([B, N])
    :param l: int
    :param normalize: whether to normalize input tensor
    :param diagonal: Tuple(int, int)
    :return: Tensor([B, N^(k+t), D]) or Tensor([B, N^(k+t-1), D)]) when diagonal is specified
    """
    bsize, n, d = A.size(0), A.size(1), A.size(-1)
    if l is None:
        l = len(A.size()) - 2
    k = len(M.size()) - l
    t = len(A.size()) - 2 - l
    assert k > 0 and l > 0 and t >= 0
    assert n == M.size(0)

    M = M.view(n ** (k + l)).unsqueeze(0).repeat(bsize, 1).view([bsize] + [n] * (k + l))  # [B, N^(k+l)]
    if normalize:
        if l == 1:
            node_mask = node_mask.view([bsize] + [1] * k + [n])
        else:
            assert l == 2
            node_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)  # [B, N, N]
            node_mask = node_mask.view([bsize] + [1] * k + [n, n])
        mask = (M * node_mask).view(bsize, n ** k, n ** l)  # [B, (N^k), (N^l)]
        vec = mask.sum(-1, keepdim=True)  # [B, (N^k)]
        M = M / vec.masked_fill_(vec == 0, 1e-5).view([bsize] + [n] * k + [1] * l)  # [B, N^(k+l)]

    if diagonal is None:
        M_ = M.view(bsize, n ** k, n ** l)  # [B, (N^k), (N^l)]
        A_ = A.reshape(bsize, n ** l, n ** t, d)  # [B, (N^l), (N^t), D]
        agg = torch.einsum('bkl,bltj->bktj', M_, A_)  # [B, (N^k), (N^t) D]
        agg = agg.reshape([bsize] + [n] * k + [n] * t + [d])  # [B, N^k, N^t, D]
        return agg
    else:
        dim1, dim2 = diagonal  # axis in [B, N^k, N^t, D]
        assert 0 < dim1 <= k < dim2 <= k + t
        # bring diagonal axis front
        # no need to do for M, as axis rotation does not change it
        M_ = M.view(bsize, n, n ** (k - 1), n ** l)  # [B, N, (N^(k-1)), (N^l)]
        A_ = rotate(A, src_dim=dim2 - k + l, tgt_dim=1)  # [B, N, N^l, N^(t-1), D]
        A_ = A_.reshape(bsize, n, n ** l, n ** (t - 1), d)  # [B, N, (N^l), (N^(t-1)), D]
        # multi-dimensional bmm
        agg = torch.einsum('bikl,biltj->biktj', M_, A_)  # [B, N, (N^(k-1)), (N^(t-1)), D]
        agg = agg.reshape([bsize] + [n] * (k + t - 1) + [d])  # [B, N^(k+t-1), D]
        # send diagonal axis back
        agg = rotate(agg, src_dim=1, tgt_dim=dim1)  # [B, N^(k+t-1), Dv]
        return agg
