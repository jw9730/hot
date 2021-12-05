import torch


def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    """Symmetrically normalize adjacency matrix
    :param adj: Tensor([N, N, 1])
    :return: Tensor([N, N, 1])
    """
    adj = adj.squeeze(-1)  # [N, N]

    # get degrees
    rowsum = adj.sum(1)  # [N,]
    d_inv_sqrt = rowsum.pow(-0.5)  # [N,]
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.

    # make diagonal normalization matrix
    d_mat_inv_sqrt = torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype) * d_inv_sqrt.unsqueeze(1)  # [N, N]

    # apply by matrix multiplication
    adj = ((adj @ d_mat_inv_sqrt).t() @ d_mat_inv_sqrt).t()  # [N, N,]

    adj = adj.unsqueeze(-1)  # [N, N, 1]
    return adj


def to_diag(A: torch.Tensor) -> torch.Tensor:
    """Note: if an entry is inf, others in the row will be set to nan, not zero
    :param A: Tensor([B, N, D])
    :return: Tensor([B, N, N, D])
    """
    assert len(A.size()) == 3
    n = A.size(1)
    eye = torch.eye(n, device=A.device, dtype=A.dtype)
    return A.unsqueeze(2) * eye.view(1, n, n, 1)


def get_diag(A: torch.Tensor) -> torch.Tensor:
    """
    :param A: Tensor([B, N, N, D])
    :return: Tensor([B, N, D])
    """
    assert len(A.size()) == 4
    return A.diagonal(dim1=1, dim2=2).transpose(2, 1)


def get_nondiag(A: torch.Tensor) -> torch.Tensor:
    """
    :param A: Tensor([B, N, N, D])
    :return: Tensor([B, N, N, D])
    """
    assert len(A.size()) == 4
    n = A.size(1)
    eye = torch.eye(n, device=A.device, dtype=A.dtype).view(1, n, n, 1)
    return A * (1 - eye)


def get_l_permutation(num_dims: int, src_dim: int, tgt_dim: int) -> list:
    """e.g., ndim = 6, src = 4, tgt = 1 -> return [0, 4, 1, 2, 3, 5]
    """
    assert tgt_dim < src_dim
    i = torch.arange(num_dims)
    pi = torch.cat((i[0:tgt_dim],
                    i[src_dim:src_dim + 1],
                    i[tgt_dim:src_dim],
                    i[src_dim + 1:]), dim=0)
    return pi.tolist()


def get_r_permutation(num_dims: int, src_dim: int, tgt_dim: int) -> list:
    """e.g., ndim = 6, src = 1, tgt = 4 -> return [0, 2, 3, 4, 1, 5]
    """
    assert src_dim < tgt_dim
    i = torch.arange(num_dims)
    pi = torch.cat((i[0:src_dim],
                    i[src_dim + 1:tgt_dim + 1],
                    i[src_dim:src_dim + 1],
                    i[tgt_dim + 1:]), dim=0)
    return pi.tolist()


def get_permutation(num_dims: int, src_dim: int, tgt_dim: int) -> list:
    if src_dim > tgt_dim:
        # left-rotation
        return get_l_permutation(num_dims, src_dim, tgt_dim)
    elif src_dim < tgt_dim:
        # right-rotation
        return get_r_permutation(num_dims, src_dim, tgt_dim)
    raise NotImplementedError('Identity rotation is inefficient')


def rotate(A: torch.Tensor, src_dim: int, tgt_dim: int) -> torch.Tensor:
    """rotate the axis of A so that src_dim goes to tgt_dim and axis between them slide by one
    """
    if src_dim == tgt_dim:
        return A
    num_dims = len(A.size())
    assert src_dim < num_dims and tgt_dim < num_dims
    return A.permute(get_permutation(num_dims, src_dim, tgt_dim))
