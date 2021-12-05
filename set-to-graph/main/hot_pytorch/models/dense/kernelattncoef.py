from typing import Union, List

import torch
import torch.nn as nn
from torch import Tensor as T

from ...batch.dense import Batch as B, batch_like
from ...utils.dense import rotate
from ..common.kernel import KernelFeatureMap
from .masksum import mask_tensor


class KernelFeatureMapWrapper(nn.Module):
    def __init__(self, feature_map: KernelFeatureMap, dim_qk, n_heads):
        super().__init__()
        self.feature_map = feature_map
        self.dim_qk_head = dim_qk // n_heads if dim_qk >= n_heads else 1

    def forward(self, G: Union[T, B], is_query=False) -> Union[T, B]:
        # this handles scaling in scaled dot-product attention
        A = G.A if isinstance(G, B) else G
        A = torch.stack(A.split(self.dim_qk_head, -1), 0)  # [H, B, N^k, D/H]
        A = self.feature_map(A, is_query)  # [H, B, N^k, D']
        if G.order > 1:
            n = A.size(2)
            loop_mask = mask_tensor(G.order, n, cast_float=False, device=A.device).view([1, 1] + [n] * G.order + [1])  # [1, 1, N^k, 1]
            A = A.masked_fill_(~loop_mask, 0)  # [H, B, N^k, D']
        A = torch.cat(A.unbind(0), -1)  # [B, N^k, D' * H]
        return batch_like(G, A, skip_masking=False)


class KernelAttnCoef(nn.Module):
    def __init__(self, ord_q, ord_k, dim_qk_head, dim_v, n_heads):
        super().__init__()
        assert ord_q in (1, 2) and ord_k in (1, 2)
        self.ord_q = ord_q
        self.ord_k = ord_k
        self.dim_qk_head = dim_qk_head  # caution: feature map dimension
        self.dim_v = dim_v
        self.dim_v_head = dim_v // n_heads if dim_v >= n_heads else 1
        self.n_heads = n_heads

    def _list_get_attn_coef(self, query_list: List[Union[T, B]], key_list: List[B]) -> List[T]:
        # compute attention coefficient simultaneously for S qk tuples
        # concatenate channels and temporarily multiply head size by S
        S = len(query_list)
        query = batch_like(query_list[0], torch.cat([query.A for query in query_list], dim=-1), skip_masking=True)
        key = batch_like(key_list[0], torch.cat([key.A for key in key_list], dim=-1), skip_masking=True)
        # scale qkv-dim and number of heads temporarily (dim_v_head doesn't change)
        cached = (self.dim_v, self.n_heads)
        self.dim_v, self.n_heads = self.dim_v * S, self.n_heads * S
        # compute attention coefficient
        alpha = self.get_attn_coef(query, key)
        # restore
        self.dim_v, self.n_heads = cached
        alpha_list = alpha.split(self.n_heads, 0)
        return alpha_list

    def get_attn_coef(self, query: Union[T, B, List[T], List[B]], key: Union[B, List[B]]) -> Union[T, List[T]]:
        if isinstance(query, list):
            return self._list_get_attn_coef(query, key)
        if isinstance(query, B):
            assert query.order == self.ord_q
        assert key.order == self.ord_k
        q_A = query.A if isinstance(query, B) else query  # [B, N^k, D']
        k_A = key.A  # [B, N^l, Dv]
        bsize, n = k_A.size(0), k_A.size(1)
        q_ = torch.stack(q_A.split(self.dim_qk_head, -1), 0)  # [H, B, N^k, D']
        k_ = torch.stack(k_A.split(self.dim_qk_head, -1), 0)  # [H, B, N^l, D']
        q_rs = q_.view(self.n_heads, bsize, n ** self.ord_q, self.dim_qk_head)  # [H, B, (N^k), D']
        k_rs = k_.view(self.n_heads, bsize, n ** self.ord_k, self.dim_qk_head)  # [H, B, (N^l), D']
        exp = torch.einsum('...kd,...ld->...kl', q_rs, k_rs)  # [H, B, (N^k), (N^l)]
        exp_sum = exp.sum(-1).unsqueeze(-1)  # [H, B, (N^k)]
        exp_sum = exp_sum.clone().masked_fill_(exp_sum == 0, 1e-5)
        alpha = exp / exp_sum  # [H, B, (N^k), (N^l)]
        if self.ord_q == 0:
            alpha = alpha.squeeze(2)
        return alpha

    def _list_forward(self, query_list: List, key_list: List[B], value_list: List[B], diagonal: tuple = None) -> Union[List[T], List[B]]:
        # compute forward simultaneously for S qkv tuples
        # concatenate channels and temporarily multiply head size by S
        S = len(query_list)
        if isinstance(query_list[0], T):
            query = torch.cat(query_list, dim=-1)
        else:
            assert isinstance(query_list[0], B)
            query = batch_like(query_list[0], torch.cat([query.A for query in query_list], dim=-1), skip_masking=True)
        key = batch_like(key_list[0], torch.cat([key.A for key in key_list], dim=-1), skip_masking=True)
        value = batch_like(value_list[0], torch.cat([value.A for value in value_list], dim=-1), skip_masking=True)
        # scale qkv-dim and number of heads temporarily (dim_v_head doesn't change)
        cached = (self.dim_v, self.n_heads)
        self.dim_v, self.n_heads = self.dim_v * S, self.n_heads * S
        # compute forward
        att = self.forward(query, key, value, diagonal)
        # restore
        self.dim_v, self.n_heads = cached
        att_list = att.A.split(self.dim_v, -1)
        return [batch_like(query, a, skip_masking=True) for a in att_list]

    def forward(self, query: Union[T, B, List[T], List[B]], key: Union[B, List[B]], value: Union[B, List[B]], diagonal: tuple = None) -> Union[B, List[B]]:
        """Compute higher-order kernelized self-attention, key-value first
        :param query: Tensor([B, Dv]) or Batch([B, N^k, Dv])
        :param key: Batch([B, N^l, Dv])
        :param value: Batch([B, N^(l+t), Dv])
        :param diagonal: axis to take diagonal after attention computation
        :return: Batch([B, N^(k+t), Dv] or [B, N^(k+t-1), Dv] if diagonal is specified)
        """
        if isinstance(value, list):
            return self._list_forward(query, key, value, diagonal)
        assert (isinstance(query, T) or isinstance(query, B)) and isinstance(key, B) and isinstance(value, B)
        if isinstance(query, B):
            assert query.order == self.ord_q
        assert key.order == self.ord_k
        q_A = query.A if isinstance(query, B) else query  # [B, N^k, Dv]
        k_A = key.A  # [B, N^l, Dv]
        v_A = value.A  # [B, N^l, N^t, Dv]
        bsize, n = k_A.size(0), k_A.size(1)
        t = value.order - self.ord_k
        q_ = torch.stack(q_A.split(self.dim_qk_head, -1), 0)  # [H, B, N^k, D']
        k_ = torch.stack(k_A.split(self.dim_qk_head, -1), 0)  # [H, B, N^l, D']
        v_ = torch.stack(v_A.split(self.dim_v_head, -1), 0)  # [H, B, N^l, N^t, Dv/H]
        q_rs = q_.view(self.n_heads, bsize, n ** self.ord_q, self.dim_qk_head)  # [H, B, (N^k), D']
        k_rs = k_.view(self.n_heads, bsize, n ** self.ord_k, self.dim_qk_head)  # [H, B, (N^l), D']
        v_rs = v_.view(self.n_heads, bsize, n ** self.ord_k, n ** t, self.dim_v_head)  # [H, B, (N^l), (N^t), Dv/H]

        # key-value aggregation
        kv_ = torch.einsum('...li,...ltj->...tij', k_rs, v_rs)  # [H, B, (N^t), D', Dv/H]

        # query-wise attention application
        if diagonal is None:
            att = torch.einsum('...ki,...tij->...ktj', q_rs, kv_)  # [H, B, (N^k), (N^t), Dv/H]
        else:
            # jointly compute and take diagonal; this reduces computation by 1/n
            dim1, dim2 = diagonal  # axis in [B, N^k, N^t, D]
            assert 0 < dim1 <= self.ord_q < dim2 <= self.ord_q + t

            # reshape key-values
            kv_ = kv_.view([self.n_heads, bsize] + [n] * t + [self.dim_qk_head, self.dim_v_head])  # [H, B, N^t, D', Dv/H]

            # bring diagonal axis front
            q_ = rotate(q_, dim1 + 1, 2)  # [H, B, N, N^(k-1), D']
            kv_ = rotate(kv_, dim2 + 1 - self.ord_q, 2)  # [H, B, N, N^(t-1), D', Dv/H]

            # reshape and compute dot product
            q_ = q_.view(self.n_heads, bsize, n, n ** (self.ord_q - 1), self.dim_qk_head)  # [H, B, N, (N^(k-1)), D']
            kv_ = kv_.view(self.n_heads, bsize, n, n ** (t - 1), self.dim_qk_head, self.dim_v_head)  # [H, B, N, (N^(t-1)), D', Dv/H]
            att = torch.einsum('...ki,...tij->...ktj', q_, kv_)  # [H, B, N, (N^(k-1)), (N^(t-1)), Dv/H]
            att = att.view([self.n_heads, bsize] + [n] * (self.ord_q + t - 1) + [self.dim_v_head])  # [H, B, N, N^(k-1), N^(t-1), Dv/H]

            # bring diagonal axis back
            att = rotate(att, 2, dim1 + 1)  # [H, B, N^(k+t-1), Dv/H]
            att = att.reshape(self.n_heads, bsize, n ** self.ord_q, n ** (t - 1), self.dim_v_head)  # [H, B, (N^k), (N^(t-1)), Dv/H]

        # normalization
        # aggregate keys and compute dot product with query
        k_sum = k_rs.sum(2)  # [H, B, D']
        qk_sum = torch.einsum('...ki,...i->...k', q_rs, k_sum)  # [H, B, (N^k)]
        qk_sum = qk_sum.view(self.n_heads, bsize, n ** self.ord_q, 1, 1)  # [H, B, (N^k), 1, 1]
        qk_sum = qk_sum.clone().masked_fill_(qk_sum == 0, 1e-5)
        att = att / qk_sum  # [H, B, (N^k), (N^(t or t-1)), Dv/H]

        # head concatenation and reshaping
        att = torch.cat(att.unbind(0), -1)  # [B, (N^k), (N^(t or t-1)), Dv]

        # mask invalid entries to 0
        if self.ord_q > 1:
            att_mask = mask_tensor(self.ord_q, n, cast_float=False, device=att.device).view(1, n ** self.ord_q, 1, 1)  # [1, (N^k), 1, 1]
            att = att.masked_fill_(~att_mask, 0)

        # reshape and return
        ord_att = self.ord_q + (t if diagonal is None else t - 1)
        att = att.view([bsize] + [n] * ord_att + [self.dim_v])  # [B, N^(k + (t or t-1)), Dv]
        return B(att, query.n_nodes)  # automatically performs masked fill with zero
