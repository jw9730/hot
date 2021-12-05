from typing import Union, List
import math

import torch
import torch.nn as nn
from torch import Tensor as T

from ...batch.dense import Batch as B, batch_like
from ...utils.dense import rotate
from .masksum import mask_tensor


class AttnCoef(nn.Module):
    def __init__(self, ord_q, ord_k, dim_qk, n_heads):
        super().__init__()
        self.ord_q = ord_q
        self.ord_k = ord_k
        self.dim_qk = dim_qk
        self.dim_qk_head = dim_qk // n_heads if dim_qk >= n_heads else 1
        self.n_heads = n_heads

    def _vector_query(self, query: T, key: B) -> T:
        """A special case where query is zeroth order batched features
        :param query: Tensor([B, Dv])
        :param key: Batch([B, N^l, Dv])
        :return: Tensor([H, B, (N^l)])
        """
        assert len(query.size()) == 2
        assert key.order == self.ord_k
        k_A = key.A   # [B, N^l, Dv]
        bsize, n = k_A.size(0), k_A.size(1)

        # make key and query tensors
        q_ = query
        k_ = k_A.reshape(bsize, n ** self.ord_k, self.dim_qk)  # [B, (N^l), Dv]
        q_ = torch.stack(q_.split(self.dim_qk_head, -1), 0)  # [H, B, Dv/H]
        k_ = torch.stack(k_.split(self.dim_qk_head, -1), 0)  # [H, B, (N^l), Dv/H]

        # make attention mask
        # should consider both batching and self-loop masking
        if self.ord_k == 1:
            att_mask = key.mask.unsqueeze(0)  # [1, B, N]
        else:
            M = mask_tensor(self.ord_k, n, cast_float=False, device=key.device).unsqueeze(0)  # [1, N^l]
            att_mask = (M & key.mask).view(1, bsize, n ** self.ord_k)  # [1, N^l], [B, N^l] -> [1, B, (N^l)]

        # compute scaled dot-product
        sdp = torch.einsum('...d,...ld->...l', q_, k_) / math.sqrt(self.dim_qk_head)  # [H, B, (N^l)]

        # apply masking and softmax
        sdp = sdp.clone().masked_fill_(~att_mask, -float('inf'))  # [H, B, (N^l)]
        alpha = torch.softmax(sdp, 2).clone()  # [H, B, (N^l)], apply normalization
        alpha = alpha.masked_fill_(~att_mask, 0)  # [H, B, (N^l)]
        return alpha

    def _list_forward(self, query_list: List, key_list: List[B]) -> List[T]:
        # compute forward simultaneously for S qkv tuples
        # concatenate channels and temporarily multiply head size by S
        S = len(query_list)
        if isinstance(query_list[0], torch.Tensor):
            query = torch.cat(query_list, dim=-1)
        else:
            query = batch_like(query_list[0], torch.cat([query.A for query in query_list], dim=-1), skip_masking=True)
        key = batch_like(key_list[0], torch.cat([key.A for key in key_list], dim=-1), skip_masking=True)
        # scale qk-dim and number of heads temporarily (dim_split doesn't change)
        cached = (self.dim_qk, self.n_heads)
        self.dim_qk, self.n_heads = self.dim_qk * S, self.n_heads * S
        # compute forward
        alpha = self.forward(query, key)
        # restore
        self.dim_qk, self.n_heads = cached
        alpha_list = alpha.split(self.n_heads, 0)
        return alpha_list

    def forward(self, query: Union[T, B, List[T], List[B]], key: Union[B, List[B]]) -> Union[T, List[T]]:
        """
        :param query: Tensor([B, Dv]) or Batch([B, N^k, Dv])
        :param key: Batch([B, N^l, Dv])
        :return: Tensor([H, B, (N^l)]) or Tensor([H, B, (N^k), (N^l)])
        """
        if isinstance(query, list):
            return self._list_forward(query, key)
        assert (isinstance(query, T) or isinstance(query, B)) and isinstance(key, B)
        assert key.order == self.ord_k
        # if query is zeroth-order, send to special case handler
        if isinstance(query, torch.Tensor) and not isinstance(query, B):
            return self._vector_query(query, key)
        assert query.order == self.ord_q

        # make key and query tensors
        q_A = query.A  # [B, N^k, Dv]
        k_A = key.A  # [B, N^l, Dv]
        bsize, n = q_A.size(0), q_A.size(1)
        assert n == k_A.size(1)
        q_ = q_A.reshape(bsize, n ** self.ord_q, self.dim_qk)  # [B, (N^k), D]
        k_ = k_A.reshape(bsize, n ** self.ord_k, self.dim_qk)  # [B, (N^l), D]
        q_ = torch.stack(q_.split(self.dim_qk_head, -1), 0)  # [H, B, (N^k), D/H]
        k_ = torch.stack(k_.split(self.dim_qk_head, -1), 0)  # [H, B, (N^l), D/H]

        # make attention mask
        q_mask = query.mask.view(bsize, n ** self.ord_q)  # [B, (N^k)]
        k_mask = key.mask.view(bsize, n ** self.ord_k)  # [B, (N^l)]
        # should consider both batching and self-loop masking
        M = mask_tensor(query.order + key.order, n, cast_float=False, device=query.device)  # [N^(k+l)]
        M = M.view(n ** query.order, n ** key.order).unsqueeze(0)  # [1, (N^k), (N^l)]
        qk_mask = q_mask.unsqueeze(2) & k_mask.unsqueeze(1)  # [B, (N^k), (N^l)]
        att_mask = (M & qk_mask).unsqueeze(0)  # [1, B, (N^k), (N^l)]

        # compute scaled dot-product
        sdp = torch.einsum('...kd,...ld->...kl', q_, k_) / math.sqrt(self.dim_qk_head)  # [H, B, (N^k), (N^l)]

        # apply masking and softmax
        sdp = sdp.clone().masked_fill_(~att_mask, -float('inf'))
        alpha = torch.softmax(sdp, 3).clone()  # [H, B, (N^k), (M^l)]
        alpha = alpha.masked_fill_(~att_mask, 0)
        return alpha


def _list_apply_attn(q_ord, k_ord, alpha_list: List[T], value_list: List[B], diagonal: tuple = None) -> Union[List[T], List[B]]:
    dim_v = value_list[0].A.size(-1)
    alpha = torch.cat(alpha_list, dim=0)  # [S * H, B, ...]
    value = batch_like(value_list[0], torch.cat([v.A for v in value_list], dim=-1), skip_masking=True)  # [B, ..., S * Dv]
    att = apply_attn(q_ord, k_ord, alpha, value, diagonal)
    if isinstance(att, torch.Tensor):
        return att.split(dim_v, -1)
    else:
        return [batch_like(att, a, skip_masking=True) for a in att.A.split(dim_v, -1)]


def apply_attn(q_ord, k_ord, alpha: Union[T, List[T]], value: Union[B, List[B]], diagonal: tuple = None) -> Union[T, B, List[T], List[B]]:
    """Apply higher-order self-attention coefficient to given value tensor
    :param q_ord: int
    :param k_ord: int
    :param alpha: Tensor([H, B, (N^l)]) or Tensor([H, B, (N^k), (N^l)])
    :param value: Batch([B, N^(l+t), Dv])
    :param diagonal: Tuple(int, int)
    :return: Tensor([B, Dv]) or Batch([B, N^(k+t), Dv] or [B, N^(k+t-1), Dv)] when diagonal is specified)
    """
    if isinstance(value, list):
        return _list_apply_attn(q_ord, k_ord, alpha, value, diagonal)
    assert isinstance(alpha, torch.Tensor) and isinstance(value, B)

    n_heads, bsize = alpha.size(0), alpha.size(1)
    n, dv = value.A.size(1), value.A.size(-1)
    k, l = q_ord, k_ord
    t = value.order - k_ord
    dim_split = dv // n_heads if dv >= n_heads else 1

    # mask invalid values to 0
    value.apply_mask(0)

    if diagonal is None:
        assert t == 0
        v_ = torch.stack(value.A.split(dim_split, -1), 0)  # [H, B, N^l, Dv/H]
        v_ = v_.reshape(n_heads, bsize, n ** l, dim_split)  # [H, B, (N^l), Dv/H]
        if k == 0:
            # handle special case (zeroth order query)
            att = torch.einsum('...l,...li->...i', alpha, v_)  # [H, B, Dv/H]
            att = torch.cat(att.unbind(0), -1)  # [B, Dv]
            return att
        else:
            att = torch.einsum('...kl,...li->...ki', alpha, v_)  # [H, B, (N^k), Dv/H]
            att = torch.cat(att.unbind(0), -1)  # [B, (N^k), Dv]
            att = att.reshape([bsize] + [n] * k + [dv])  # [B, N^k, Dv]
            return B(att, value.n_nodes)  # automatically performs masked fill with zero
    else:
        # jointly compute and take diagonal; this reduces memory footprint by 1/n
        dim1, dim2 = diagonal  # axis in [B, N^k, M^t, D]
        assert 0 < dim1 <= k < dim2 <= k + t
        a_ = alpha.reshape([n_heads, bsize] + [n] * (k + l))  # [H, B, N^k, N^l]
        v_ = torch.stack(value.A.split(dim_split, -1), 0)  # [H, B, N^l, N^t, Dv/H]

        # bring diagonal axis front
        a_ = rotate(a_, src_dim=dim1 + 1, tgt_dim=2)  # [H, B, N, N^(k-1), N^l]
        a_ = a_.reshape(n_heads, bsize, n, n ** (k-1), n ** l)  # [H, B, N, (N^(k-1)), (N^l)]
        v_ = rotate(v_, src_dim=dim2 + 1 - k + l, tgt_dim=2)  # [H, B, N, N^l, N^(t-1), Dv/H]
        v_ = v_.reshape(n_heads, bsize, n, n ** l, n ** (t-1), dim_split)  # [H, B, N, (N^l), (N^(t-1)), Dv/H]

        # multi-dimensional bmm
        att = torch.einsum('...nkl,...nltj->...nktj', a_, v_)  # [H, B, N, (N^(k-1)), (N^(t-1)), Dv/H]
        att = torch.cat(att.unbind(0), -1)  # [B, N, (N^(k-1)), (N^(t-1)), Dv]
        att = att.reshape([bsize] + [n] * (k + t - 1) + [dv])  # [B, N, N^(k-1), N^(t-1), Dv]

        # send diagonal axis back
        att = rotate(att, src_dim=1, tgt_dim=dim1)  # [B, N^(k+t-1), Dv]
        return B(att, value.n_nodes)  # automatically performs masked fill with zero
