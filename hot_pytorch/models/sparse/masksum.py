import torch


def batch_mask(mask1: torch.BoolTensor, mask2: torch.BoolTensor) -> torch.BoolTensor:
    """
    :param mask1: BoolTensor([B, |E1|])
    :param mask2: BoolTensor([B, |E2|])
    :return: BoolTensor([B, |E1|, |E2|])
    """
    mask1 = mask1.unsqueeze(2)  # [B, |E1|, 1]
    mask2 = mask2.unsqueeze(1)  # [B, 1, |E2|]
    return mask1 & mask2  # [B, |E1|, |E2|]


def diagonalization_mask(idx1: torch.LongTensor, idx2: torch.LongTensor) -> torch.BoolTensor:
    """
    :param idx1: LongTensor([B, |E1|, 1])
    :param idx2: LongTensor([B, |E2|, 1])
    :return: BoolTensor([B, |E1|, |E2|])
    """
    assert idx1.size(-1) == idx2.size(-1) == 1
    return idx1 == idx2.squeeze(-1).unsqueeze(1)  # [B, 1, |E2|]


def count_unique(mat: torch.Tensor):
    """Count unique entries in each row
    :param mat: LongTensor([N, M])
    :return: LongTensor([N,])
    """
    assert len(mat.size()) == 2
    mat = mat.clone()
    n, m = mat.size()
    n_unique = torch.zeros(n, device=mat.device)
    for ptr in range(m):
        cur = mat[:, ptr].unsqueeze(1)  # [N, 1]
        is_cur = torch.logical_and(mat == cur, cur != -1)  # [N, M]
        exists = torch.sum(is_cur, 1) > 0  # [N, 1]
        n_unique += exists.float()
        mat[is_cur] = -1
    assert (mat + 1 == 0).all()
    return n_unique.long()


def loop_exclusion_mask(idx1: torch.LongTensor, idx2: torch.LongTensor) -> torch.BoolTensor:
    """
    :param idx1: LongTensor([B, |E1|, k])
    :param idx2: LongTensor([B, |E2|, l])
    :return: BoolTensor([B, |E1|, |E2|])
    """
    bsize = idx1.size(0)
    k, l = idx1.size(-1), idx2.size(-1)
    assert 1 <= k <= 2 and 1 <= l <= 2

    # pairwise index tensor
    idx1 = idx1.unsqueeze(2).expand(bsize, idx1.size(1), idx2.size(1), k)  # [B, |E1|, |E2|, k]
    idx2 = idx2.unsqueeze(1).expand(bsize, idx1.size(1), idx2.size(1), l)  # [B, |E1|, |E2|, l]
    idx = torch.cat((idx1, idx2), dim=-1)  # [B, |E1|, |E2|, k+l]

    # exclusion by unique element counting
    idx_mat = idx.view(-1, k + l)  # [B*|E1|*|E2|, k+l]
    n_unique = count_unique(idx_mat)  # [B*|E1|*|E2|,]
    n_unique = n_unique.view(bsize, idx.size(1), idx.size(2))  # [B, |E1|, |E2|]
    return n_unique == k + l  # [B, |E1|, |E2|]
