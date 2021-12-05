from typing import Union, Tuple, List
import math

import torch
import torch.nn as nn
from torch import sparse_coo_tensor as coo
from torch import Tensor as T

import numpy as np

from ...batch.sparse import Batch as B, batch_like
from ...utils.set import to_batch
from .masksum import batch_mask, loop_exclusion_mask as loop_mask


class AttnCoef(nn.Module):
    def __init__(self, ord_q, ord_k, dim_qk, n_heads):
        super().__init__()
        self.ord_q = ord_q
        self.ord_k = ord_k
        self.dim_qk = dim_qk
        self.dim_qk_head = dim_qk // n_heads if dim_qk >= n_heads else 1
        self.n_heads = n_heads

    def _vector_query(self, query: T, key: B, get_exp: bool) -> Union[T, Tuple[T, T]]:
        """A special case where query is zeroth order batched features
        :param query: Tensor([B, D])
        :param key: Batch([B, |E|, l], [B, |E|, D])
        :param get_exp:
        :return: Tensor([H, B, |E|])
        """
        assert len(query.size()) == 2
        assert key.order == self.ord_k

        # make key and query tensors
        q_ = torch.stack(query.split(self.dim_qk_head, -1), 0)  # [B, D] -> [H, B, D/H]
        k_ = torch.stack(key.values.split(self.dim_qk_head, -1), 0)  # [B, |E|, D] -> [H, B, |E|, D/H]

        # make attention mask
        if self.ord_k == 1:
            att_mask = key.mask.unsqueeze(0)  # [1, B, |E|]
        else:
            assert self.ord_k == 2
            # should consider both batching and self-loop masking
            nondiag_mask = key.indices[..., 0] != key.indices[..., 1]  # [B, |E|]
            att_mask = (nondiag_mask & key.mask).unsqueeze(0)  # [1, B, |E|]

        # compute scaled dot-product
        sdp = torch.einsum('...d,...ed->...e', q_, k_) / math.sqrt(self.dim_qk_head)  # [H, B, |E|]

        # apply masking and softmax
        sdp = sdp.clone().masked_fill_(~att_mask, -float('inf'))  # [H, B, |E|]
        alpha = torch.softmax(sdp, 2).clone()  # [H, B, |E|]
        alpha = alpha.masked_fill_(~att_mask, 0)
        if get_exp:
            exp = torch.exp(sdp - torch.max(sdp)).clone()
            exp = exp.masked_fill_(~att_mask, 0)
            return alpha, exp
        return alpha

    def _list_forward(self, query_list: List, key_list: List[B], get_exp: bool) -> Union[List[T], Tuple[List[T], List[T]]]:
        # compute forward simultaneously for S qkv tuples
        # concatenate channels and temporarily multiply head size by S
        S = len(query_list)
        if isinstance(query_list[0], torch.Tensor):
            query = torch.cat(query_list, dim=-1)
        else:
            query = batch_like(query_list[0], torch.cat([query.values for query in query_list], dim=-1), skip_masking=True)
        key = batch_like(key_list[0], torch.cat([key.values for key in key_list], dim=-1), skip_masking=True)
        # scale qk-dim and number of heads temporarily (dim_split doesn't change)
        cached = (self.dim_qk, self.n_heads)
        self.dim_qk, self.n_heads = self.dim_qk * S, self.n_heads * S
        # compute forward
        if get_exp:
            alpha, exp = self.forward(query, key, True)
            # restore
            self.dim_qk, self.n_heads = cached
            return alpha.split(self.n_heads, 0), exp.split(self.n_heads, 0)
        else:
            alpha = self.forward(query, key, False)
            # restore
            self.dim_qk, self.n_heads = cached
            return alpha.split(self.n_heads, 0)

    def forward(self, query: Union[T, B, List[T], List[B]], key: Union[B, List[B]], get_exp=False) -> Union[T, Tuple[T, T], List[T], List[Tuple[T, T]]]:
        """
        :param query: Tensor([B, D]) or Batch([B, |Eq|, k], [B, |Eq|, D])
        :param key: Batch([B, |Ek|, l], [B, |Ek|, D])
        :param get_exp: return unnormalized values exp(scaled dot-product) for unmatched sparse aggregation
        :return: Tensor([H, B, |Ek|]) or Tensor([H, B, |Eq|, |Ek|])
        """
        if isinstance(query, list):
            return self._list_forward(query, key, get_exp)
        assert (isinstance(query, T) or isinstance(query, B)) and isinstance(key, B)
        assert key.order == self.ord_k
        # if query is zeroth-order, send to special case handler
        if isinstance(query, T) and not isinstance(query, B):
            return self._vector_query(query, key, get_exp)
        assert query.order == self.ord_q

        # make key and query tensors
        q_v = query.values  # [B, |Eq|, D]
        k_v = key.values  # [B, |Ek|, D]
        q_ = torch.stack(q_v.split(self.dim_qk_head, -1), 0)  # [H, B, |Eq|, D/H]
        k_ = torch.stack(k_v.split(self.dim_qk_head, -1), 0)  # [H, B, |Ek|, D/H]

        # make attention mask
        n = max(query.n_nodes)
        bsize = q_v.size(0)
        # should consider both batching and self-loop masking
        node_idx = torch.arange(n, device=q_v.device)[None, :, None].expand(bsize, n, 1)  # [B, N, 1]
        q_i = query.indices if self.ord_q > 1 else node_idx  # [B, |Eq|, 1]
        k_i = key.indices if self.ord_k > 1 else node_idx  # [B, |Ek|, 1]
        att_mask = batch_mask(query.mask, key.mask) & loop_mask(q_i, k_i)  # [B, |Eq|, |Ek|]
        att_mask = att_mask.unsqueeze(0)  # [1, B, |Eq|, |Ek|]

        # compute scaled dot-product
        sdp = torch.einsum('...qd,...kd->...qk', q_, k_) / math.sqrt(self.dim_qk_head)  # [H, B, |Eq|, |Ek|]

        # apply masking and softmax
        sdp = sdp.clone().masked_fill_(~att_mask, -float('inf'))
        alpha = torch.softmax(sdp, 3).clone()  # [H, B, |Eq|, |Ek|]
        alpha = alpha.masked_fill_(~att_mask, 0)
        if get_exp:
            exp = torch.exp(sdp - torch.max(sdp)).clone()
            exp = exp.masked_fill_(~att_mask, 0)
            return alpha, exp
        return alpha


def _list_apply_attn(query: Union[T, B], k_ord, alpha_list: List[T], value_list: List[B], diagonal: tuple = None) -> Union[List[T], List[B]]:
    dim_v = value_list[0].values.size(-1)
    alpha = torch.cat(alpha_list, dim=0)  # [S * H, B, ...]
    value = batch_like(value_list[0], torch.cat([v.values for v in value_list], dim=-1), skip_masking=True)  # [B, ..., S * Dv]
    att = apply_attn(query, k_ord, alpha, value, diagonal)
    if isinstance(att, torch.Tensor):
        return att.split(dim_v, -1)
    else:
        return [batch_like(att, a, skip_masking=True) for a in att.values.split(dim_v, -1)]


def apply_attn(query: Union[T, B], k_ord, alpha: Union[T, List[T]], value: Union[B, List[B]], diagonal: tuple = None) -> Union[T, B, List[T], List[B]]:
    """Apply higher-order self-attention coefficient to given value tensor
    :param query: Tensor([B, Dv]) or Batch([B, |Eq|, k], [B, |Eq|, Dv])
    :param k_ord:
    :param alpha: Tensor([H, B, |Ek|]) or Tensor([H, B, |Eq|, |Ek|])
    :param value: Batch([B, |Ev|, l+t], [B, |Ev|, Dv]), l + t <= 2
    :param diagonal: Tuple(int, int)
    :return: Tensor([B, Dv]) or Tensor([B, |Eq|, Dv])
    """
    if isinstance(value, list):
        return _list_apply_attn(query, k_ord, alpha, value, diagonal)
    assert isinstance(alpha, torch.Tensor) and isinstance(value, B)

    n_heads, bsize = alpha.size(0), alpha.size(1)
    dv = value.values.size(-1)
    k = query.order if isinstance(query, B) else 0
    l = k_ord
    t = value.order - l
    dim_v_head = dv // n_heads if dv >= n_heads else 1

    # mask invalid values to 0
    value.apply_mask(0)

    if diagonal is None:
        assert t == 0
        # we can simply aggregate because Ek == Ev
        v_ = torch.stack(value.values.split(dim_v_head, -1), 0)  # [B, |Ev|, Dv] -> [H, B, |Ev|, Dv/H]
        if k == 0:
            # handle special case (zeroth order query)
            att = torch.einsum('...i,...id->...d', alpha, v_)  # [H, B, Dv/H]
            att = torch.cat(att.unbind(0), -1)  # [B, Dv]
            return att
        else:
            att = torch.einsum('...ji,...id->...jd', alpha, v_)  # [H, B, |Eq|, Dv/H]
            att = torch.cat(att.unbind(0), -1)  # [B, |Eq|, Dv]
            return batch_like(query, att, skip_masking=False)
    else:
        # jointly compute and take diagonal; this reduces memory footprint by 1/n
        # in this case, aggregation can be sparse; softmax -> aggregation leads to over-suppression
        # to handle this, it is assumed that alpha is not normalized (exp is given)
        dim1, dim2 = diagonal
        # always key = set, value = graph
        assert l == 1 and t == 1
        # case-by-case handling
        n = alpha.size(3)
        exp = alpha  # [H, B, |Eq|, N]
        if k == 1:
            # query = set, key = set, value = graph
            assert (dim1, dim2) == (1, 2)
            # Oi = Σj(exp)ij*Vji / Σj(exp)ij = Σj(exp)ij*V.t()ij / Σj(exp)ij = Σj(exp*V.t())ij / Σj(exp)ij
            # 1. transpose value
            v_ = torch.stack(value.values.split(dim_v_head, -1), 0)  # [B, |E|, Dv] -> [H, B, |E|, Dv/H]
            v_t_idx = value.indices[..., [1, 0]].clone().clamp_(0)  # [B, |E|, 2]
            v_t_idx_1d = v_t_idx[..., 0] * n + v_t_idx[..., 1]  # [B, |E|]
            v_t_idx_1d = v_t_idx_1d[None, ...].expand(n_heads, bsize, v_t_idx_1d.size(1))  # [H, B, |E|]

            # 2. elementwise multiplication with unnormalized attention coefficient
            exp_gather = exp.flatten(2, 3).gather(2, v_t_idx_1d)    # [H, B, N * N] -> [H, B, |E|]
            exp_v_prod = exp_gather.unsqueeze(-1) * v_  # [H, B, |E|, Dv/H]
            exp_v_prod = torch.cat(exp_v_prod.unbind(0), -1)  # [B, |E|, Dv]
            exp_concat = torch.cat([exp_v_prod, exp_gather.permute(1, 2, 0)], dim=-1)  # [B, |E|, Dv + H]

            # 3. row-wise summation
            exp_row_idx = v_t_idx[..., 0] + value.node_ofs[:, None]  # [B, |E|]
            batch_row_idx = exp_row_idx[value.mask].unsqueeze(0)  # [1, |E|]
            batch_exp_val = exp_concat[value.mask]  # [|E|, Dv + H]
            exp_sp = coo(batch_row_idx, batch_exp_val, size=(sum(value.n_nodes), dv + n_heads))  # [N, Dv + H]
            exp_row_sum = exp_sp.coalesce().to_dense()  # [N, Dv + H]

            # 4. normalization
            exp_v_prod = exp_row_sum[:, :dv]  # [N, Dv]
            exp_v_prod = torch.stack(exp_v_prod.split(dim_v_head, -1), 0)  # [N, Dv] -> [H, N, Dv/H]
            exp_sum = exp_row_sum[:, dv:].t().unsqueeze(-1)  # [N, H]
            exp_sum = exp_sum.clone().masked_fill_(exp_sum == 0, 1e-5)
            att_unmasked = exp_v_prod / exp_sum

            att_unmasked = torch.cat(att_unmasked.unbind(0), -1)  # [N, Dv]
            att = to_batch(att_unmasked, value.n_nodes, value.node_mask, 0)  # [B, N, Dv], [B, N]
            return batch_like(query, att, skip_masking=False)
        else:
            # query = graph, key = set, value = graph
            assert k == 2
            v_ = torch.stack(value.values.split(dim_v_head, -1), 0)  # [B, |E|, Dv] -> [H, B, |E|, Dv/H]
            # 1. get index for summation (k) and diagonalization (i or j)
            v_sum_idx = value.indices[..., 0]  # [B, |E|] (k)
            v_diag_idx = value.indices[..., 1]  # [B, |E|] (j or i)
            if (dim1, dim2) == (2, 3):
                # Oij = Σk(exp)ijk*Vkj / Σk(exp)ijk
                exp_diag_idx = query.indices[..., 1]  # [B, |E|] (j)
            else:
                assert (dim1, dim2) == (1, 3)
                # Oij = Σk(exp)ijk*Vki / Σk(exp)ijk
                exp_diag_idx = query.indices[..., 0]  # [B, |E|] (i)

            # 2. expand and mask attention coefficients for joint diagonalization and summation
            # note: expansion causes a memory bottleneck
            e_q = exp.size(2)
            e_v = v_sum_idx.size(1)
            sum_idx = v_sum_idx.clone().clamp_(0)  # [B, |E|]
            sum_idx = sum_idx.view(1, bsize, 1, e_v).expand(n_heads, bsize, e_q, e_v)  # [H, B, |E|, |E|]
            exp_gather = exp.gather(3, sum_idx)  # [H, B, |E|, |E|]
            diag_mask = exp_diag_idx.unsqueeze(-1) == v_diag_idx.unsqueeze(1)  # [B, |E|, |E|]
            diag_mask = diag_mask[None, ...].expand(n_heads, bsize, e_q, e_v)
            exp_gather = exp_gather.masked_fill_(~diag_mask, 0)  # [H, B, |E|, |E|]

            # 3. application and normalization
            exp = torch.einsum('...ij,...jd->...id', exp_gather, v_)  # [H, B, |E|, Dv/H]
            exp_sum = exp_gather.sum(-1).unsqueeze(-1)  # [H, B, |E|, 1]
            exp_sum = exp_sum.clone().masked_fill_(exp_sum == 0, 1e-5)
            att = exp / exp_sum  # [H, B, |E|, Dv/H]

            att = torch.cat(att.unbind(0), -1)  # [B, |E|, Dv]
            return batch_like(query, att, skip_masking=False)
