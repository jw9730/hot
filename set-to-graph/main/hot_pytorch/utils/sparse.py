from typing import Tuple

import torch

from .set import to_batch


def normalize_adj(adj_i: torch.LongTensor, adj_v: torch.Tensor, n: int) -> Tuple[torch.LongTensor, torch.Tensor]:
    """Symmetrically normalize adjacency matrix
    :param adj_i: LongTensor([2, |E|])
    :param adj_v: Tensor([|E|, 1])
    :param n: int
    :return: LongTensor([2, |E|]), Tensor([|E|, 1])
    """
    adj = torch.sparse_coo_tensor(adj_i, adj_v.squeeze(-1), size=(n, n))  # [N, N]

    # get degrees
    rowsum = torch.sparse.sum(adj, 1).to_dense()  # [N,]
    d_inv_sqrt = rowsum.pow(-0.5)  # [N,]
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.

    # make diagonal normalization matrix
    diag_i = torch.arange(n, device=adj.device).unsqueeze(0).repeat(2, 1)  # [2, N]
    d_mat_inv_sqrt = torch.sparse_coo_tensor(diag_i, d_inv_sqrt, size=(n, n))  # [N, N]

    # apply by matrix multiplication
    adj = torch.sparse.mm(torch.sparse.mm(adj, d_mat_inv_sqrt).t(), d_mat_inv_sqrt).t().coalesce()  # [N, N]

    adj_i = adj.indices()  # [2, |E|]
    adj_v = adj.values().unsqueeze(-1)  # [|E|, 1]
    return adj_i, adj_v


def to_diag(indices: torch.LongTensor, values: torch.Tensor, mask: torch.BoolTensor, node_mask: torch.BoolTensor) -> torch.Tensor:
    """
    :param indices: LongTensor([B, |E|, 2])
    :param values: Tensor([B, N, D])
    :param mask: BoolTensor([B, |E|])
    :param node_mask: BoolTensor([B, N])
    :return: Tensor([B, |E|, D])
    """
    diag_mask = (indices[..., 0] == indices[..., 1]) & mask  # [B, |E|]
    # caution: to reduce overhead, we assume that indices are organized nicely:
    # (1) contains all diagonal indices without omit
    # (2) diagonal indices are ordered
    # e.g. for a data with 3 nodes,
    # row idx: [1 2 3 x x x]
    # col idx: [1 2 3 y y y]
    # if else, one must use the following code:
    #     # gather with row index and mask out non-diagonal entries
    #     indices = indices[..., :1].repeat(1, 1, values.size(-1))  # [B, |E|, D]
    #     D_ = torch.gather(values, 1, indices)  # [B, |E|, D]
    #     D_ = D_.masked_fill_(~diag_mask.unsqueeze(-1), 0)  # [B, |E|, D]
    D = torch.zeros(indices.size(0), indices.size(1), values.size(-1), device=values.device, dtype=values.dtype)  # [B, |E|, D]
    D[diag_mask] = values[node_mask]
    return D


def get_diag(indices: torch.LongTensor, values: torch.Tensor, n_nodes: list, mask: torch.BoolTensor,
             node_mask: torch.BoolTensor, node_ofs: torch.LongTensor) -> torch.Tensor:
    """
    :param indices: LongTensor([B, |E|, 2])
    :param values: Tensor([B, |E|, D])
    :param n_nodes: [n1, ..., nb]
    :param mask: BoolTensor([B, |E|])
    :param node_mask: BoolTensor([B, N])
    :param node_ofs: LongTensor([B,])
    :return: Tensor([B, N, D]), BoolTensor([B, N])
    """
    diag_mask = (indices[..., 0] == indices[..., 1]) & mask  # [B, |E|]
    v = values[diag_mask]  # [|Ed|, D]
    # caution: to reduce overhead, we assume that indices are organized nicely:
    # (1) contains all diagonal indices without omit
    # (2) diagonal indices are ordered
    # e.g. for a data with 3 nodes,
    # row idx: [1 2 3 x x x]
    # col idx: [1 2 3 y y y]
    # This makes |Ed| == N, and the condition can be tested with the below code:
    #     # get batched index by adding offset
    #     batch_i = indices + node_ofs[:, None, None]  # [B, |E|, 2]
    #     i = batch_i[diag_mask][None, :, 0]  # [1, |Ed|]
    #     assert i.size(1) == sum(n_nodes) and (i[0] == torch.arange(sum(n_nodes), device=i.device)).all()
    # if else, one must use the following code:
    #     D = torch.sparse_coo_tensor(i, v, size=(sum(n_nodes), v.size(-1)))  # [N, D]
    #     D = D.coalesce().to_dense()  # [N, D]
    #     return to_batch(D, n_nodes, node_mask, 0)
    return to_batch(v, n_nodes, node_mask, 0)


def get_nondiag(indices: torch.LongTensor, values: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
    """
    :param indices: LongTensor([B, |E|, 2])
    :param values: Tensor([B, |E|, D])
    :param mask: BoolTensor([B, |E|])
    :return: Tensor([B, |E|, D])
    """
    nondiag_mask = (indices[..., 0] != indices[..., 1]) & mask  # [B, |E|]
    ND = values.clone().masked_fill_(~nondiag_mask.unsqueeze(-1), 0)
    return ND


def get_transpose_info(indices: torch.LongTensor, mask: torch.BoolTensor, chunk_size=1000):
    """
    :param indices: LongTensor([B, |E|, 2])
    :param mask: BoolTensor([B, |E|])
    :param chunk_size: for memory footprint control
    :return: LongTensor([B, |E|]), BoolTensor([B, |E|])
    """
    indices_t = indices[..., [1, 0]]  # [B, |E|, 2]
    n_chunks = int(indices.size(1) / chunk_size)
    with torch.no_grad():
        if n_chunks > 1:
            indices_chunks = indices.split(chunk_size, 1)
            transpose_indices_list = []
            exist_mask_list = []
            for indices_chunk in indices_chunks:
                M_chunk = (indices_chunk.unsqueeze(2) == indices_t.unsqueeze(1)).all(-1)  # [B, |E|', |E|]
                transpose_indices_list.append(M_chunk.float().argmax(2).detach())  # [B, |E|']
                exist_mask_list.append(M_chunk.any(2).detach())  # [B, |E|']
                del M_chunk
            transpose_indices = torch.cat(transpose_indices_list, 1)  # [B, |E|]
            exist_mask = torch.cat(exist_mask_list, 1) & mask
        else:
            M = (indices.unsqueeze(2) == indices_t.unsqueeze(1)).all(-1)  # [B, |E|, |E|]
            transpose_indices = M.float().argmax(2)  # [B, |E|]
            exist_mask = M.any(2) & mask  # [B, |E|]
    return transpose_indices, exist_mask


def do_transpose(values: torch.Tensor, transpose_indices: torch.LongTensor, exist_mask: torch.BoolTensor):
    # assumption: input tensor is coalesced
    T = torch.gather(values, 1, transpose_indices.unsqueeze(-1).repeat(1, 1, values.size(-1)))  # [B, |E|, D]
    T = T.masked_fill_(~exist_mask.unsqueeze(-1), 0)
    return T
