import torch
import torch.nn as nn

from ...batch.dense import Batch as B, batch_like, t, v2d, d, add_batch
from ..common.mudrop import MuDropout
from .linear import Linear
from .kernelattncoef import KernelFeatureMapWrapper, KernelAttnCoef


class KernelSelfAttn(nn.Module):
    def __init__(self, ord_in, ord_out, dim_in, dim_v, dim_qk, n_heads, cfg='default', dropout=0., drop_mu=0., feature_map=None):
        super().__init__()
        assert cfg in ('default', 'local')
        self.is_local = cfg == 'local'
        self.ord_in = ord_in
        self.ord_out = ord_out
        self.dim_in = dim_in
        self.dim_v = dim_v
        self.dim_qk = dim_qk
        self.n_heads = n_heads
        self.feature_map = KernelFeatureMapWrapper(feature_map, dim_qk, n_heads)
        self.feat_dim = feature_map.num_features
        if (ord_in, ord_out) == (1, 0):
            raise ValueError('Kernel gives no asymptotic improvement. Use softmax instead')
        elif (ord_in, ord_out) == (1, 1):
            n_qk1, n_v = 2, 1
            self.fc_1 = Linear(1, 1, dim_in, dim_qk * n_qk1 + dim_in, cfg='light')
            self.att_1_1 = KernelAttnCoef(1, 1, self.feat_dim, dim_v, n_heads)
        elif (ord_in, ord_out) == (1, 2):
            n_qk1, n_qk2, n_v = 3, 1, 2
            self.fc_1 = Linear(1, 1, dim_in, dim_qk * n_qk1, cfg='light')
            self.fc_2 = Linear(1, 2, dim_in, dim_qk * n_qk2 + dim_in, cfg='light')
            self.att_1_1 = KernelAttnCoef(1, 1, self.feat_dim, dim_v, n_heads)
            self.att_2_1 = KernelAttnCoef(2, 1, self.feat_dim, dim_v, n_heads)
        elif (ord_in, ord_out) == (2, 0):
            raise ValueError('Kernel gives no asymptotic improvement. Use softmax instead')
        elif (ord_in, ord_out) == (2, 1):
            n_qk1, n_qk2, n_v = 7, 1, 4
            self.fc_1 = Linear(2, 1, dim_in, dim_qk * n_qk1 + dim_in, cfg='light')
            self.fc_2 = Linear(2, 2, dim_in, dim_qk * n_qk2, cfg='light')
            self.att_1_1 = KernelAttnCoef(1, 1, self.feat_dim, dim_v, n_heads)
            self.att_1_2 = KernelAttnCoef(1, 2, self.feat_dim, dim_v, n_heads)
        elif (ord_in, ord_out) == (2, 2):
            n_qk1, n_qk2, n_v = 12, 8, 10
            self.fc_1 = Linear(2, 1, dim_in, dim_qk * n_qk1, cfg='light')
            self.fc_2 = Linear(2, 2, dim_in, dim_qk * n_qk2 + dim_in, cfg='light')
            self.att_1_1 = KernelAttnCoef(1, 1, self.feat_dim, dim_v, n_heads)
            self.att_2_1 = KernelAttnCoef(2, 1, self.feat_dim, dim_v, n_heads)
            self.att_1_2 = KernelAttnCoef(1, 2, self.feat_dim, dim_v, n_heads)
            self.att_2_2 = KernelAttnCoef(2, 2, self.feat_dim, dim_v, n_heads)
        else:
            raise NotImplementedError
        self.fc_v = nn.Linear(dim_in, dim_v * n_v)
        self.fc_o = nn.Linear(dim_v * n_v, dim_in)
        self.reset_vo_parameters()
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.mu_dropout = MuDropout(p=drop_mu)

    def reset_vo_parameters(self):
        nn.init.xavier_normal_(self.fc_v.weight)
        nn.init.xavier_normal_(self.fc_o.weight)
        nn.init.constant_(self.fc_v.bias, 0.)
        nn.init.constant_(self.fc_o.bias, 0.)

    def get_qk_list(self, G: B):
        v_list = G.A.split(self.feat_dim * self.n_heads, -1)
        return [batch_like(G, v, skip_masking=True) for v in v_list]

    def get_v_list(self, G: B):
        v = batch_like(G, self.fc_v(G.A), skip_masking=False)
        A_list = v.A.split(self.dim_v, -1)
        return [batch_like(G, a, skip_masking=True) for a in A_list]

    def combine_att(self, G: B, att_list):
        att = self.fc_o(self.dropout(torch.cat(self.mu_dropout([a.A for a in att_list]), -1)))
        return batch_like(G, att, skip_masking=False)

    def _1_to_1(self, G: B):
        # compute query, key and value
        h_1 = self.fc_1(G)  # [B, N, (2+1)D]
        non_att = batch_like(h_1, h_1.A[..., -self.dim_in:], skip_masking=True)  # [B, N, D]
        q_1 = batch_like(h_1, h_1.A[..., :self.dim_qk], skip_masking=True)  # [B, N, D]
        k_1 = batch_like(h_1, h_1.A[..., self.dim_qk:self.dim_qk * 2], skip_masking=True)  # [B, N, D]
        v_1_list = self.get_v_list(G)  # List([B, N, Dv])
        # kernel feature map
        q_1 = self.feature_map(q_1, is_query=True)  # [B, N, D']
        k_1 = self.feature_map(k_1, is_query=False)  # [B, N, D']

        # set -> set
        att_1 = self.att_1_1(q_1, k_1, v_1_list[0])  # [B, N, D]
        # combine
        att = batch_like(G, self.fc_o(self.dropout(att_1.A)), skip_masking=False)
        return add_batch(non_att, att)

    def _1_to_2(self, G: B):
        # compute query, key and value
        h_1 = self.fc_1(G)  # [B, N, 3D]
        q_1 = batch_like(h_1, h_1.A[..., :self.dim_qk], skip_masking=True)  # [B, N, D]
        k_1 = batch_like(h_1, h_1.A[..., self.dim_qk:], skip_masking=True)  # [B, N, 2D]
        h_2 = self.fc_2(G)  # [B, N, N, (1+1)D]
        non_att = batch_like(h_2, h_2.A[..., -self.dim_in:], skip_masking=True)  # [B, N, N, D]
        q_2 = batch_like(h_2, h_2.A[..., :self.dim_qk], skip_masking=True)  # [B, N, N, D]
        v_1_list = self.get_v_list(G)  # List([B, N, Dv])
        q_1 = self.feature_map(q_1, is_query=True)  # [B, N, D']
        q_2 = self.feature_map(q_2, is_query=True)  # [B, N, N, D']
        k_1 = self.feature_map(k_1, is_query=False)  # [B, N, 2D']
        k_1_list = self.get_qk_list(k_1)  # List([B, N, D'] * 2)

        # set -> graph
        att_1 = self.att_2_1(q_2, k_1_list[0], v_1_list[0])  # [B, N, N, D]
        # set -> set
        att_2 = v2d(self.att_1_1(q_1, k_1_list[1], v_1_list[1]), q_2.mask)  # [B, N, D] -> [B, N, N, D]
        # combine
        att_list = [att_1, att_2]
        att = self.combine_att(q_2, att_list)
        return add_batch(non_att, att)

    def _2_to_1(self, G: B):
        # compute query, key and value
        h_1 = self.fc_1(G)  # [B, N, (7+1)D]
        non_att = batch_like(h_1, h_1.A[..., -self.dim_in:], skip_masking=True)  # [B, N, D]
        q_1 = batch_like(h_1, h_1.A[..., :self.dim_qk * 4], skip_masking=True)  # [B, N, 4D]
        k_1 = batch_like(h_1, h_1.A[..., self.dim_qk * 4:self.dim_qk * 7], skip_masking=True)  # [B, N, 3D]
        k_2 = self.fc_2(G)  # [B, |E|, D]
        v_2_list = self.get_v_list(G)  # List([B, |E|, Dv])
        # kernel feature map
        q_1 = self.feature_map(q_1, is_query=True)  # [B, N, 4D']
        k_1 = self.feature_map(k_1, is_query=False)  # [B, N, 3D']
        k_2 = self.feature_map(k_2, is_query=False)  # [B, |E|, D']
        q_1_list = self.get_qk_list(q_1)  # List([B, N, D'])
        k_1_list = self.get_qk_list(k_1)  # List([B, N, D'])

        # graph -> set
        att_1, att_2 = self.att_1_1(q_1_list[0:2], k_1_list[0:2], [v_2_list[0], t(v_2_list[1])], diagonal=(1, 2))  # [B, N, Dv]
        att_list = [att_1, att_2]
        if not self.is_local:
            # set -> set
            att_3 = self.att_1_1(q_1_list[2], k_1_list[2], d(v_2_list[2]))  # [B, N, Dv]
            # graph -> set
            att_4 = self.att_1_2(q_1_list[3], k_2, v_2_list[3])  # [B, N, Dv]
            att_list += [att_3, att_4]
        # combine
        att = self.combine_att(q_1_list[0], att_list)
        return add_batch(non_att, att)

    def _2_to_2(self, G: B):
        # compute query, key and value
        h_1 = self.fc_1(G)  # [B, N, 12D]
        q_1 = batch_like(h_1, h_1.A[..., :self.dim_qk * 4], skip_masking=True)  # [B, N, 4D]
        k_1 = batch_like(h_1, h_1.A[..., self.dim_qk * 4:], skip_masking=True)  # [B, N, 8D]
        h_2 = self.fc_2(G)  # [B, |E|, (8+1)D]
        non_att = batch_like(h_2, h_2.A[..., -self.dim_in:], skip_masking=True)  # [B, N, D]
        q_2 = batch_like(h_2, h_2.A[..., :self.dim_qk * 6], skip_masking=True)  # [B, |E|, 6D]
        k_2 = batch_like(h_2, h_2.A[..., self.dim_qk * 6:self.dim_qk * 8], skip_masking=True)  # [B, |E|, 2D]
        v_2_list = self.get_v_list(G)  # [B, |E|, 10D]
        # kernel feature map
        q_1 = self.feature_map(q_1, is_query=True)  # [B, N, 4D']
        q_2 = self.feature_map(q_2, is_query=True)  # [B, |E|, 8D']
        k_1 = self.feature_map(k_1, is_query=False)  # [B, N, 6D']
        k_2 = self.feature_map(k_2, is_query=False)  # [B, |E|, 2D']
        q_1_list = self.get_qk_list(q_1)  # List([B, N, D'])
        q_2_list = self.get_qk_list(q_2)  # List([B, |E|, D'])
        k_1_list = self.get_qk_list(k_1)  # List([B, N, D'])
        k_2_list = self.get_qk_list(k_2)  # List([B, |E|, D'])

        # graph -> set
        att_1, att_2 = self.att_1_1(q_1_list[0:2], k_1_list[0:2], [v_2_list[0], t(v_2_list[1])], diagonal=(1, 2))  # [B, N, D] -> [B, |E|, D]
        att_1 = v2d(att_1, G.mask)
        att_2 = v2d(att_2, G.mask)
        # graph -> graph
        att_3, att_5 = self.att_2_1(q_2_list[0:2], k_1_list[2:4], [v_2_list[2], t(v_2_list[3])], diagonal=(2, 3))
        att_4, att_6 = self.att_2_1(q_2_list[2:4], k_1_list[4:6], [t(v_2_list[4]), v_2_list[5]], diagonal=(1, 3))
        att_list = [att_1, att_2, att_3, att_4, att_5, att_6]
        if not self.is_local:
            # set -> set
            att_7 = v2d(self.att_1_1(q_1_list[2], k_1_list[6], d(v_2_list[6])), G.mask)  # [B, N, D] -> [B, N, N, D]
            # graph -> set
            att_8 = v2d(self.att_1_2(q_1_list[3], k_2_list[0], v_2_list[7]), G.mask)  # [B, N, D] -> [B, N, N, D]
            # set -> graph
            att_9 = self.att_2_1(q_2_list[4], k_1_list[7], d(v_2_list[8]))  # [B, N, D] -> [B, |E|, D]
            # graph -> graph
            att_10 = self.att_2_2(q_2_list[5], k_2_list[1], v_2_list[9])
            att_list += [att_7, att_8, att_9, att_10]
        # combine
        att = self.combine_att(G, att_list)
        return add_batch(non_att, att)

    def forward(self, G: B):
        assert G.order == self.ord_in

        if (self.ord_in, self.ord_out) == (1, 1):
            G_att = self._1_to_1(G)
        elif (self.ord_in, self.ord_out) == (1, 2):
            G_att = self._1_to_2(G)
        elif (self.ord_in, self.ord_out) == (2, 1):
            G_att = self._2_to_1(G)
        elif (self.ord_in, self.ord_out) == (2, 2):
            G_att = self._2_to_2(G)
        else:
            raise NotImplementedError('Currently supports up to second-order invariance only')

        if self.ord_out > 0:
            assert G_att.order == self.ord_out
        else:
            assert isinstance(G_att, torch.Tensor)
        return G_att
