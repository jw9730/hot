from typing import Union, List

import torch
import torch.nn as nn
from torch import sparse_coo_tensor as coo
from torch import Tensor as T

from ...batch.sparse import Batch as B, batch_like
from ...utils.set import to_batch
from ..common.kernel import KernelFeatureMap


class KernelFeatureMapWrapper(nn.Module):
    def __init__(self, feature_map: KernelFeatureMap, dim_qk, n_heads):
        super().__init__()
        self.feature_map = feature_map
        self.dim_qk_head = dim_qk // n_heads if dim_qk >= n_heads else 1

    def forward(self, G: B, is_query=False) -> B:
        # this handles scaling in scaled dot-product attention
        v = torch.stack(G.values.split(self.dim_qk_head, -1), 0)  # [H, B, |E|, D_H]
        v = self.feature_map(v, is_query)  # [H, B, |E|, D']
        if G.order > 1:
            loop_mask = G.indices[..., 0] != G.indices[..., 1]  # [B, |E|]
            loop_mask = loop_mask[None, :, :, None]  # [1, B, |E|, 1]
            v = v.masked_fill_(~loop_mask, 0)  # [S, B, |E|, D']
        v = torch.cat(v.unbind(0), -1)  # [B, |E|, D' * H]
        return batch_like(G, v, skip_masking=False)


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

    def _list_get_attn_coef(self, query_list: List[B], key_list: List[B]) -> List[T]:
        # compute attention coefficient simultaneously for S qk tuples
        # concatenate channels and temporarily multiply head size by S
        S = len(query_list)
        query = batch_like(query_list[0], torch.cat([query.values for query in query_list], dim=-1), skip_masking=True)
        key = batch_like(key_list[0], torch.cat([key.values for key in key_list], dim=-1), skip_masking=True)
        # scale qkv-dim and number of heads temporarily (dim_v_head doesn't change)
        cached = (self.dim_v, self.n_heads)
        self.dim_v, self.n_heads = self.dim_v * S, self.n_heads * S
        # compute attention coefficient
        alpha = self.get_attn_coef(query, key)
        # restore
        self.dim_v, self.n_heads = cached
        alpha_list = alpha.split(self.n_heads, 0)
        return alpha_list

    def get_attn_coef(self, query: Union[B, List[B]], key: Union[B, List[B]]) -> Union[T, List[T]]:
        if isinstance(query, list):
            return self._list_get_attn_coef(query, key)
        if isinstance(query, B):
            assert query.order == self.ord_q
        assert key.order == self.ord_k
        q_ = query.values  # [B, |Eq|, D' * H]
        k_ = key.values  # [B, |Ek|, D' * H]
        q_ = torch.stack(q_.split(self.dim_qk_head, -1), 0)  # [H, B, |Eq|, D']
        k_ = torch.stack(k_.split(self.dim_qk_head, -1), 0)  # [H, B, |Ek|, D']
        exp = torch.einsum('...kd,...ld->...kl', q_, k_)  # [H, B, |Eq|, |Ek|]
        exp_sum = exp.sum(-1).unsqueeze(-1)  # [H, B, |Eq|, 1]
        exp_sum = exp_sum.clone().masked_fill_(exp_sum == 0, 1e-5)
        alpha = exp / exp_sum  # [H, B, |Eq|, |Ek|]
        return alpha

    def _full_key_value_sum_and_apply_query(self, q_, k_, v_):
        # 1. key-value aggregation: assume Ek == Ev
        k_ = torch.stack(k_.split(self.dim_qk_head, -1), 0)  # [H, B, |E|, D']
        v_ = torch.stack(v_.split(self.dim_v_head, -1), 0)  # [H, B, |E|, Dv/H]
        kv_ = torch.einsum('...li,...lj->...ij', k_, v_)  # [H, B, D', Dv/H]

        # 2. query-wise attention application
        q_ = torch.stack(q_.split(self.dim_qk_head, -1), 0)  # [H, B, |Eq|, D']
        att = torch.einsum('...ei,...ij->...ej', q_, kv_)  # [H, B, |Eq|, Dv/H]

        # 3. normalization: aggregate keys and compute dot product with query
        k_sum = k_.sum(2)  # [H, B, D']
        qk_sum = torch.einsum('...id,...d->...i', q_, k_sum).unsqueeze(-1)  # [H, B, |Eq|, 1]
        qk_sum = qk_sum.clone().masked_fill_(qk_sum == 0, 1e-5)
        att = att / qk_sum  # [H, B, |Eq|, Dv/H]

        att = torch.cat(att.unbind(0), -1)  # [B, |Eq|, Dv]
        return att

    def _concat_batch(self, query: B, key: B, value: B):
        # to avoid overhead in gather(), operate in concatenated batch representation
        q_ = query.values[query.mask]  # [|Eq|, D' * H]
        k_ = key.values[key.mask]  # [N, D' * H]
        v_ = value.values[value.mask]  # [|E|, Dv]
        q_ = torch.stack(q_.split(self.dim_qk_head, -1), 0)  # [H, |Eq|, D']
        k_ = torch.stack(k_.split(self.dim_qk_head, -1), 0)  # [H, N, D']
        v_ = torch.stack(v_.split(self.dim_v_head, -1), 0)  # [H, |E|, Dv/H]
        return q_, k_, v_

    def _partial_key_value_sum(self, k_, v_, value: B):
        # 1. key-value aggregation
        bsize, n = len(value.n_nodes), sum(value.n_nodes)
        # must jointly take diagonal with query index, such that (KV)j = Σi(Ki Vij^T)
        v_idx = value.indices + value.node_ofs.view(bsize, 1, 1)  # [B, |E|, 2]
        v_idx = v_idx[value.mask]  # [|E|, 2]
        v_row_idx = v_idx[:, :1]  # [|E|, 1] (i)
        v_col_idx = v_idx[:, 1:]  # [|E|, 1] (j)
        # expand key to match values
        v_row_idx = v_row_idx[None, ...].expand(self.n_heads, v_row_idx.size(0), self.dim_qk_head)  # [H, |E|, D']
        k_gather = torch.gather(k_, 1, v_row_idx)  # [H, |E|, D']
        # key-value elementwise product
        kv_ = torch.einsum('...i,...j->...ij', k_gather, v_)  # [H, |E|, D', Dv/H]
        kv_ = torch.cat(kv_.unbind(0), -1).flatten(1, 2)  # [|E|, D', Dv] -> [|E|, D' * Dv]
        k_gather = torch.cat(k_gather.unbind(0), -1)  # [|E|, D' * H]
        kv_k_cat = torch.cat([kv_, k_gather], dim=-1)  # [|E|, D' * Dv + D' * H]
        # column-wise summation via coalescing
        kv_k_cat_dim = self.dim_qk_head * self.dim_v + self.dim_qk_head * self.n_heads
        kv_k_cat = coo(v_col_idx.t(), kv_k_cat, size=(n, kv_k_cat_dim)).coalesce().to_dense()  # [N, D' * Dv + D' * H]
        return kv_k_cat

    def _partial_apply_set_query(self, q_, kv_cat, query: B):
        n = sum(query.n_nodes)
        # separate to summed key-value pairs and summed keys
        kv_ = kv_cat[..., :self.dim_qk_head * self.dim_v]  # [N, D' * Dv]
        k_sum = kv_cat[..., self.dim_qk_head * self.dim_v:]  # [N, D' * H]
        kv_ = kv_.view(n, self.dim_qk_head, self.dim_v)  # [N, D', Dv]
        kv_ = torch.stack(kv_.split(self.dim_v_head, -1), 0)  # [H, N, D', Dv/H]
        k_sum = torch.stack(k_sum.split(self.dim_qk_head, -1), 0)  # [H, N, D']

        # 2. query-wise attention application
        att = torch.einsum('...i,...ij->...j', q_, kv_)  # [H, N, D'], [H, N, D', Dv/H] -> [H, N, Dv/H]

        # 3. normalization
        qk_sum = torch.einsum('...i,...i->...', q_, k_sum).unsqueeze(-1)  # [H, N, 1]
        qk_sum = qk_sum.clone().masked_fill_(qk_sum == 0, 1e-5)
        att = att / qk_sum  # [H, N, Dv/H]

        att = torch.cat(att.unbind(0), -1)  # [N, Dv]
        att = to_batch(att, query.n_nodes, query.node_mask)  # [B, N, Dv]
        return att

    def _partial_apply_graph_query(self, q_, kv_cat, query: B, dim1):
        bsize = len(query.n_nodes)
        # expand to query size
        q_idx = query.indices + query.node_ofs.view(bsize, 1, 1)  # [B, |E|, 2]
        q_idx = q_idx[query.mask]  # [|E|, 2]
        if dim1 == 2:
            # Oij = Qij^T Σk(Kk Vkj^T) / Qij^T Σk(Kk)
            q_idx = q_idx[:, 1:]  # [|E|, 1], col idx
        else:
            # Oij = Qij^T Σk(Kk Vki^T) / Qij^T Σk(Kk)
            q_idx = q_idx[:, :1]  # [|E|, 1], row idx
        q_idx = q_idx.expand(q_idx.size(0), self.dim_qk_head * self.dim_v + self.dim_qk_head * self.n_heads)  # [|E|, D' * Dv + D' * H]
        kv_cat = torch.gather(kv_cat, 0, q_idx)  # [|E|, D' * Dv + D' * H]
        # separate to summed key-value pairs and summed keys
        kv_ = kv_cat[..., :self.dim_qk_head * self.dim_v]  # [|E|, D' * Dv]
        k_sum = kv_cat[..., self.dim_qk_head * self.dim_v:]  # [|E|, D' * H]
        kv_ = kv_.view(kv_cat.size(0), self.dim_qk_head, self.dim_v)  # [|E|, D', Dv]
        kv_ = torch.stack(kv_.split(self.dim_v_head, -1), 0)  # [H, |E|, D', Dv/H]
        k_sum = torch.stack(k_sum.split(self.dim_qk_head, -1), 0)  # [H, |E|, D']

        # 2. query-wise attention application
        att = torch.einsum('...i,...ij->...j', q_, kv_)  # [H, |E|, Dv/H]

        # 3. normalization
        qk_sum = torch.einsum('...i,...i->...', q_, k_sum).unsqueeze(-1)  # [H, |E|, 1]
        qk_sum = qk_sum.clone().masked_fill_(qk_sum == 0, 1e-5)
        att = att / qk_sum  # [H, |E|, Dv/H]

        att = torch.cat(att.unbind(0), -1)  # [|E|, Dv]
        att = to_batch(att, query.n_edges, query.mask)  # [B, |E|, Dv]
        return att

    def _list_forward(self, query_list: List, key_list: List[B], value_list: List[B], diagonal: tuple = None) -> List[B]:
        # compute forward simultaneously for S qkv tuples
        # concatenate channels and temporarily multiply head size by S
        S = len(query_list)
        if isinstance(query_list[0], T):
            query = torch.cat(query_list, dim=-1)
        else:
            assert isinstance(query_list[0], B)
            query = batch_like(query_list[0], torch.cat([query.values for query in query_list], dim=-1), skip_masking=True)
        key = batch_like(key_list[0], torch.cat([key.values for key in key_list], dim=-1), skip_masking=True)
        value = batch_like(value_list[0], torch.cat([value.values for value in value_list], dim=-1), skip_masking=True)
        # scale qkv-dim and number of heads temporarily (dim_v_head doesn't change)
        cached = (self.dim_v, self.n_heads)
        self.dim_v, self.n_heads = self.dim_v * S, self.n_heads * S
        # compute forward
        att = self.forward(query, key, value, diagonal)
        # restore
        self.dim_v, self.n_heads = cached
        att_list = att.values.split(self.dim_v, -1)
        return [batch_like(query, a, skip_masking=True) for a in att_list]

    def forward(self, query: Union[B, List[B]], key: Union[B, List[B]], value: Union[B, List[B]], diagonal: tuple = None) -> Union[B, List[B]]:
        """Compute higher-order kernelized self-attention, key-value first
        :param query: Batch([B, |Eq|, k], [B, |Eq|, D' * H])
        :param key: Batch([B, |Ek|, l], [B, |Ek|, D' * H])
        :param value: Batch([B, |Ev|, l+t], [B, |Ev|, Dv])
        :param diagonal: axis to take diagonal after attention computation
        :return: Batch([B, |Ev|, l+t or l+t-1 if diagonal is specified], [B, |Ev|, Dv])
        """
        if isinstance(value, list):
            return self._list_forward(query, key, value, diagonal)
        assert isinstance(query, B) and isinstance(key, B) and isinstance(value, B)
        assert query.order == self.ord_q and key.order == self.ord_k
        if diagonal is None:
            assert value.order == self.ord_k
            q_, k_, v_ = query.values, key.values, value.values  # [B, |Eq|, D' * H], [B, |Ek|, D' * H], [B, |Ev|, Dv]
            att = self._full_key_value_sum_and_apply_query(q_, k_, v_)
        else:
            # jointly compute and take diagonal
            dim1, dim2 = diagonal
            assert self.ord_k == 1 and value.order == 2
            assert (self.ord_q, dim1, dim2) in [(1, 1, 2), (2, 2, 3), (2, 1, 3)]
            q_, k_, v_ = self._concat_batch(query, key, value)
            kv_k_cat = self._partial_key_value_sum(k_, v_, value)
            if self.ord_q == 1:
                # Oj = Qj^T Σi(Ki Vij^T) / Qj^T Σi(Ki)
                att = self._partial_apply_set_query(q_, kv_k_cat, query)
            else:
                # dim1 = 1: Oij = Qij^T Σk(Kk Vki^T) / Qij^T Σk(Kk)
                # dim1 = 2: Oij = Qij^T Σk(Kk Vkj^T) / Qij^T Σk(Kk)
                att = self._partial_apply_graph_query(q_, kv_k_cat, query, dim1)
        if self.ord_q == 1:
            return batch_like(query, att, skip_masking=False)
        else:
            loop_mask = query.indices[..., 0] != query.indices[..., 1]  # [B, |E|]
            loop_mask = loop_mask[..., None]  # [B, |E|, 1]
            att = att.masked_fill_(~loop_mask, 0)  # [B, |E|, Dv]
            return batch_like(query, att, skip_masking=False)
