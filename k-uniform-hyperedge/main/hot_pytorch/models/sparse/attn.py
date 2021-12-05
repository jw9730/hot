import torch
import torch.nn as nn

from ...batch.sparse import Batch as B, batch_like, t, v2d, d, add_batch
from ..common.mudrop import MuDropout
from .linear import Linear
from .attncoef import AttnCoef, apply_attn


class SelfAttn(nn.Module):
    def __init__(self, ord_in, ord_out, dim_in, dim_v, dim_qk, n_heads, cfg='default', dropout=0., drop_mu=0.):
        super().__init__()
        assert cfg in ('default', 'local')
        self.is_local = cfg == 'local'
        self.ord_in = ord_in
        self.ord_out = ord_out
        self.dim_in = dim_in
        self.dim_v = dim_v
        self.dim_qk = dim_qk
        self.n_heads = n_heads
        if (ord_in, ord_out) == (1, 0):
            n_qk0, n_qk1, n_v = 1, 1, 1
            self.fc_0 = Linear(1, 0, dim_in, dim_qk * n_qk0 + dim_in, cfg='light')
            self.fc_1 = Linear(1, 1, dim_in, dim_qk * n_qk1, cfg='light')
            self.att_0_1 = AttnCoef(0, 1, dim_qk, n_heads)
        elif (ord_in, ord_out) == (1, 1):
            n_qk1, n_v = 2, 1
            self.fc_1 = Linear(1, 1, dim_in, dim_qk * n_qk1 + dim_in, cfg='light')
            self.att_1_1 = AttnCoef(1, 1, dim_qk, n_heads)
        elif (ord_in, ord_out) == (1, 2):
            raise NotImplementedError('Sparse set-to-graph is inefficient; use a dense layer or specialized layers')
        elif (ord_in, ord_out) == (2, 0):
            n_qk0, n_qk1, n_qk2, n_v = 2, 1, 1, 2
            self.fc_0 = Linear(2, 0, dim_in, dim_qk * n_qk0 + dim_in, cfg='light')
            self.fc_1 = Linear(2, 1, dim_in, dim_qk * n_qk1, cfg='light')
            self.fc_2 = Linear(2, 2, dim_in, dim_qk * n_qk2, cfg='light')
            self.att_0_1 = AttnCoef(0, 1, dim_qk, n_heads)
            self.att_0_2 = AttnCoef(0, 2, dim_qk, n_heads)
        elif (ord_in, ord_out) == (2, 1):
            n_qk1, n_qk2, n_v = 7, 1, 4
            self.fc_1 = Linear(2, 1, dim_in, dim_qk * n_qk1 + dim_in, cfg='light')
            self.fc_2 = Linear(2, 2, dim_in, dim_qk * n_qk2, cfg='light')
            self.att_1_1 = AttnCoef(1, 1, dim_qk, n_heads)
            self.att_1_2 = AttnCoef(1, 2, dim_qk, n_heads)
        elif (ord_in, ord_out) == (2, 2):
            n_qk1, n_qk2, n_v = 12, 8, 10
            self.fc_1 = Linear(2, 1, dim_in, dim_qk * n_qk1, cfg='light')
            self.fc_2 = Linear(2, 2, dim_in, dim_qk * n_qk2 + dim_in, cfg='light')
            self.att_1_1 = AttnCoef(1, 1, dim_qk, n_heads)
            self.att_2_1 = AttnCoef(2, 1, dim_qk, n_heads)
            self.att_1_2 = AttnCoef(1, 2, dim_qk, n_heads)
            self.att_2_2 = AttnCoef(2, 2, dim_qk, n_heads)
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
        v_list = G.values.split(self.dim_qk, -1)
        return [batch_like(G, v, skip_masking=True) for v in v_list]

    def get_v_list(self, G: B):
        v = batch_like(G, self.fc_v(G.values), skip_masking=False)
        v_list = v.values.split(self.dim_v, -1)
        return [batch_like(G, v, skip_masking=True) for v in v_list]

    def combine_att(self, G: B, att_list):
        att = self.fc_o(self.dropout(torch.cat(self.mu_dropout([a.values for a in att_list]), -1)))
        return batch_like(G, att, skip_masking=False)

    def _1_to_0(self, G: B):
        # compute query, key and value
        h_0 = self.fc_0(G)  # [B, (1+1)D]
        non_att = h_0[..., -self.dim_in:]  # [B, D]
        q_0 = h_0[..., :self.dim_qk]  # [B, D]
        k_1 = self.fc_1(G)  # [B, N, D]
        v_1_list = self.get_v_list(G)  # List([B, N, Dv])
        # get attention tensors
        alpha_0_1 = self.att_0_1(q_0, k_1)  # [H, B, N]

        # set -> vec
        att_1 = apply_attn(q_0, 1, alpha_0_1, v_1_list[0])  # [B, Dv]
        # combine
        att = self.fc_o(self.dropout(att_1))  # [B, Dv]
        return add_batch(non_att, att)

    def _1_to_1(self, G: B):
        # compute query, key and value
        h_1 = self.fc_1(G)  # [B, N, (2+1)D]
        non_att = batch_like(h_1, h_1.values[..., -self.dim_in:], skip_masking=True)  # [B, N, D]
        q_1 = batch_like(h_1, h_1.values[..., :self.dim_qk], skip_masking=True)  # [B, N, D]
        k_1 = batch_like(h_1, h_1.values[..., self.dim_qk:self.dim_qk * 2], skip_masking=True)  # [B, N, D]
        v_1_list = self.get_v_list(G)  # List([B, N, Dv])
        # get attention tensors
        alpha_1_1 = self.att_1_1(q_1, k_1)  # [H, B, N, N]

        # set -> set
        att_1 = apply_attn(q_1, 1, alpha_1_1, v_1_list[0])  # [B, N, Dv]
        # combine
        att = batch_like(G, self.fc_o(self.dropout(att_1.values)), skip_masking=False)
        return add_batch(non_att, att)

    def _2_to_0(self, G: B):
        # compute query, key and value
        h_0 = self.fc_0(G)  # [B, (2+1)D]
        non_att = h_0[..., -self.dim_in:]  # [B, D]
        q_0_list = h_0[..., :self.dim_qk * 2].split(self.dim_qk, -1)  # List([B, D] * 2)
        k_1 = self.fc_1(G)  # [B, N, D]
        k_2 = self.fc_2(G)  # [B, |E|, D]
        v_2_list = self.get_v_list(G)  # List([B, N, Dv])
        # get attention tensors
        alpha_0_1 = self.att_0_1(q_0_list[0], k_1)  # [H, B, N]
        alpha_0_2 = self.att_0_2(q_0_list[1], k_2)  # [H, B, |E|]

        # set -> vec
        att_1 = apply_attn(q_0_list[0], 1, alpha_0_1, d(v_2_list[0]))  # [B, Dv]
        # graph -> vec
        att_2 = apply_attn(q_0_list[0], 2, alpha_0_2, v_2_list[1])  # [B, Dv]
        # combine
        att_list = [att_1, att_2]
        att = self.fc_o(self.dropout(torch.cat(att_list, -1)))
        return add_batch(non_att, att)

    def _2_to_1(self, G: B):
        # compute query, key and value
        h_1 = self.fc_1(G)  # [B, N, (7+1)D]
        non_att = batch_like(h_1, h_1.values[..., -self.dim_in:], skip_masking=True)  # [B, N, D]
        q_1_list = self.get_qk_list(batch_like(h_1, h_1.values[..., :self.dim_qk * 4], skip_masking=True))  # List([B, N, D] * 4)
        k_1_list = self.get_qk_list(batch_like(h_1, h_1.values[..., self.dim_qk * 4:self.dim_qk * 7], skip_masking=True))  # List([B, N, D] * 3)
        k_2 = self.fc_2(G)  # [B, |E|, D]
        v_2_list = self.get_v_list(G)  # List([B, |E|, Dv])
        # get attention tensors
        alpha_1_1_list, exp_1_1_list = self.att_1_1(q_1_list[:3], k_1_list[:3], get_exp=True)  # List([H, B, N, N])
        alpha_1_2 = self.att_1_2(q_1_list[3], k_2)  # [H, B, N, |E|]

        # graph -> set
        att_1, att_2 = apply_attn(q_1_list[0], 1, exp_1_1_list[0:2], [v_2_list[0], t(v_2_list[1])], diagonal=(1, 2))  # [B, N, D]
        att_list = [att_1, att_2]
        if not self.is_local:
            # set -> set
            att_3 = apply_attn(q_1_list[0], 1, alpha_1_1_list[2], d(v_2_list[2]))  # [B, N, D]
            # graph -> set
            att_4 = apply_attn(q_1_list[0], 2, alpha_1_2, v_2_list[3])  # [B, N, D]
            att_list += [att_3, att_4]
        # combine
        att = self.combine_att(q_1_list[0], att_list)
        return add_batch(non_att, att)

    def _2_to_2(self, G: B):
        # compute query, key and value
        h_1 = self.fc_1(G)  # [B, N, 7D]
        q_1_list = self.get_qk_list(batch_like(h_1, h_1.values[..., :self.dim_qk * 4], skip_masking=True))  # List([B, N, D] * 4)
        k_1_list = self.get_qk_list(batch_like(h_1, h_1.values[..., self.dim_qk * 4:], skip_masking=True))  # List([B, N, D] * 8)
        h_2 = self.fc_2(G)  # [B, N, N, (8+1)D]
        non_att = batch_like(h_2, h_2.values[..., -self.dim_in:], skip_masking=True)  # [B, |E|, D]
        q_2_list = self.get_qk_list(batch_like(h_2, h_2.values[..., :self.dim_qk * 6], skip_masking=True))  # List([B, |E|, D] * 6)
        k_2_list = self.get_qk_list(batch_like(h_2, h_2.values[..., self.dim_qk * 6:self.dim_qk * 8], skip_masking=True))  # List([B, |E|, D] * 2)
        v_2_list = self.get_v_list(G)  # List([B, |E|, Dv])
        # get attention tensors
        alpha_1_1_list, exp_1_1_list = self.att_1_1(q_1_list[:3], k_1_list[:3], get_exp=True)  # List([H, B, N, N])
        alpha_1_2 = self.att_1_2(q_1_list[3], k_2_list[0])  # [H, B, N, |E|]
        alpha_2_1_list, exp_2_1_list = self.att_2_1(q_2_list[:5], k_1_list[3:], get_exp=True)  # List([H, B, |E|, N])
        alpha_2_2 = self.att_2_2(q_2_list[5], k_2_list[1])  # [H, B, |E|, |E|]

        # graph -> set
        att_1, att_2 = apply_attn(q_1_list[0], 1, exp_1_1_list[0:2], [v_2_list[0], t(v_2_list[1])], diagonal=(1, 2))  # [B, N, Dv] -> [B, |E|, Dv]
        att_1 = v2d(G, att_1)
        att_2 = v2d(G, att_2)
        # graph -> graph
        att_3, att_5 = apply_attn(q_2_list[0], 1, exp_2_1_list[:2], [v_2_list[2], t(v_2_list[3])], diagonal=(2, 3))
        att_4, att_6 = apply_attn(q_2_list[0], 1, exp_2_1_list[2:4], [t(v_2_list[4]), v_2_list[5]], diagonal=(1, 3))
        att_list = [att_1, att_2, att_3, att_4, att_5, att_6]
        if not self.is_local:
            # set -> set
            att_7 = v2d(G, apply_attn(q_1_list[0], 1, alpha_1_1_list[2], d(v_2_list[6])))  # [B, N, D] -> [B, |E|, D]
            # graph -> set
            att_8 = v2d(G, apply_attn(q_1_list[0], 2, alpha_1_2, v_2_list[7]))  # [B, N, D] -> [B, |E|, D]
            # set -> graph
            att_9 = apply_attn(q_2_list[0], 1, alpha_2_1_list[4], d(v_2_list[8]))  # [B, N, D] -> [B, |E|, D]
            # graph -> graph
            att_10 = apply_attn(q_2_list[0], 2, alpha_2_2, v_2_list[9])
            att_list += [att_7, att_8, att_9, att_10]
        # combine
        att = self.combine_att(G, att_list)
        return add_batch(non_att, att)

    def forward(self, G: B):
        assert G.order == self.ord_in

        if (self.ord_in, self.ord_out) == (1, 0):
            G_att = self._1_to_0(G)
        elif (self.ord_in, self.ord_out) == (1, 1):
            G_att = self._1_to_1(G)
        elif (self.ord_in, self.ord_out) == (2, 0):
            G_att = self._2_to_0(G)
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
