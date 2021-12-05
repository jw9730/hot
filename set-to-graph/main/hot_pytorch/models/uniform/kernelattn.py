import torch
import torch.nn as nn

from .linear import Linear


class KernelSelfAttn(nn.Module):
    def __init__(self, ord_in, ord_out, dim_in, dim_v, dim_qk, n_heads, cfg='default', dropout=0., drop_mu=0., feature_map=None):
        super().__init__()
        assert cfg == 'default'
        self.ord_in = ord_in
        self.ord_out = ord_out
        self.dim_in = dim_in
        self.dim_v = dim_v
        self.dim_qk = dim_qk
        self.n_heads = n_heads
        self.feature_map = feature_map
        self.dim_qk_head = dim_qk // n_heads if dim_qk >= n_heads else 1
        self.dim_v_head = dim_v // n_heads if dim_v >= n_heads else 1
        self.feat_dim = feature_map.num_features
        if (ord_in, ord_out) == (1, 1):
            n_qk1, n_v = 2, 1
            self.fc_1 = Linear(1, 1, dim_in, dim_qk * n_qk1 + dim_in, bias=True, cfg='light')
        elif ord_in == 1 and ord_out > 1:
            n_qk1, n_qkk, n_v = 1, 1, 1
            self.fc_1 = Linear(1, 1, dim_in, dim_qk * n_qk1, bias=True, cfg='light')
            self.fc_k = Linear(1, ord_out, dim_in, dim_qk * n_qkk + dim_in, bias=True, cfg='light')
        else:
            raise NotImplementedError('This extension is only for 1->k-uniform')
        self.fc_v = nn.Linear(dim_in, dim_v * n_v)
        self.fc_o = nn.Linear(dim_v * n_v, dim_in)
        self.reset_vo_parameters()
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def reset_vo_parameters(self):
        nn.init.xavier_normal_(self.fc_v.weight)
        nn.init.xavier_normal_(self.fc_o.weight)
        nn.init.constant_(self.fc_v.bias, 0.)
        nn.init.constant_(self.fc_o.bias, 0.)

    def _1_to_1(self, x: torch.Tensor) -> torch.Tensor:
        h_1 = self.fc_1(x)  # [N, 3D]
        q_1 = h_1[:, :self.dim_qk]  # [N, D]
        k_1 = h_1[:, self.dim_qk:self.dim_qk*2]  # [N, D]
        non_att = h_1[:, -self.dim_in:]  # [N, D]
        q_1 = torch.stack(q_1.split(self.dim_qk_head, -1), 0)  # [H, N, D/H]
        k_1 = torch.stack(k_1.split(self.dim_qk_head, -1), 0)  # [H, N, D/H]
        v_1 = torch.stack(self.fc_v(x).split(self.dim_v_head, -1), 0)  # [H, N, Dv/H]
        # kernel feature map
        q_1 = self.feature_map(q_1, is_query=True)  # [H, N, D']
        k_1 = self.feature_map(k_1, is_query=False)  # [H, N, D']
        # set -> set
        kv = torch.einsum('hni,hnj->hij', k_1, v_1)  # [H, D', Dv/H]
        att = torch.einsum('hni,hij->hnj', q_1, kv)  # [H, N, Dv/H]
        k_sum = k_1.sum(1)  # [H, D']
        qk_sum = torch.einsum('hni,hi->hn', q_1, k_sum)[:, :, None]  # [H, N, 1]
        qk_sum = qk_sum.clone().masked_fill_(qk_sum == 0, 1e-5)
        att = att / qk_sum
        att = torch.cat(att.unbind(0), -1)  # [N, Dv]
        return non_att + self.fc_o(self.dropout(att))  # [B, Dv]

    def _1_to_k(self, x: torch.Tensor, indices: torch.LongTensor) -> torch.Tensor:
        k_1 = self.fc_1(x)  # [N, D]
        h_k = self.fc_k(x, indices)  # [B, 2D]
        q_k = h_k[:, :self.dim_qk]  # [B, D]
        non_att = h_k[:, -self.dim_in:]  # [B, D]
        q_k = torch.stack(q_k.split(self.dim_qk_head, -1), 0)  # [H, B, D/H]
        k_1 = torch.stack(k_1.split(self.dim_qk_head, -1), 0)  # [H, N, D/H]
        v_1 = torch.stack(self.fc_v(x).split(self.dim_v_head, -1), 0)  # [H, N, Dv/H]
        # kernel feature map
        q_k = self.feature_map(q_k, is_query=True)  # [H, B, D']
        k_1 = self.feature_map(k_1, is_query=False)  # [H, N, D']
        # set -> k-edge
        kv = torch.einsum('hni,hnj->hij', k_1, v_1)  # [H, D', Dv/H]
        att = torch.einsum('hni,hij->hnj', q_k, kv)  # [H, B, Dv/H]
        k_sum = k_1.sum(1)  # [H, D']
        qk_sum = torch.einsum('hni,hi->hn', q_k, k_sum)[:, :, None]  # [H, B, 1]
        qk_sum = qk_sum.clone().masked_fill_(qk_sum == 0, 1e-5)
        att = att / qk_sum
        att = torch.cat(att.unbind(0), -1)  # [B, Dv]
        return non_att + self.fc_o(self.dropout(att))  # [B, Dv]

    def forward(self, x: torch.Tensor, indices: torch.LongTensor = None) -> torch.Tensor:
        """
        :param x: Tensor([N, D])
        :param indices: Tensor([B, k]) or None
        :return: Tensor([B, D']) or Tensor([N, D'])
        """
        assert len(x.size()) == 2
        if self.ord_out == 1:
            assert indices is None
            x = self._1_to_1(x)
        else:
            assert len(indices.size()) == 2 and indices.size(1) == self.ord_out
            x = self._1_to_k(x, indices)
        return x
