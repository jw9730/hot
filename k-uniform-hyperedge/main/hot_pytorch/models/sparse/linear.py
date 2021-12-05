from typing import Union

import torch
import torch.nn as nn

from ...batch.sparse import Batch as B, batch_like, do_transpose, d, nd
from ...utils.sparse import to_diag, get_nondiag
from .masksum import diagonalization_mask as diag_mask, loop_exclusion_mask as loop_mask, batch_mask


def _normalize(values, mask):
    vec = mask.sum(-1, keepdim=True)
    return values / vec.clone().masked_fill_(vec == 0, 1e-5)


class _Weight(nn.Module):
    def __init__(self, ord_in, ord_out, dim_in, dim_out, cfg='default', normalize=False):
        super().__init__()
        assert cfg in ('default', 'light')
        self.ord_in = ord_in
        self.ord_out = ord_out
        self.cfg = cfg
        self.normalize = normalize
        if (ord_in, ord_out) == (0, 0):
            n_w = {'default': 1, 'light': 1}[cfg]
        elif (ord_in, ord_out) == (1, 0):
            n_w = {'default': 1, 'light': 1}[cfg]
        elif (ord_in, ord_out) == (1, 1):
            n_w = {'default': 2, 'light': 1}[cfg]
        elif (ord_in, ord_out) == (1, 2):
            n_w = {'default': 5, 'light': 3}[cfg]
        elif (ord_in, ord_out) == (2, 0):
            n_w = {'default': 2, 'light': 2}[cfg]
        elif (ord_in, ord_out) == (2, 1):
            n_w = {'default': 5, 'light': 1}[cfg]
        elif (ord_in, ord_out) == (2, 2):
            n_w = {'default': 15, 'light': 5}[cfg]
        else:
            raise NotImplementedError('Currently supports up to second-order only')
        self.weight = nn.Parameter(torch.Tensor(dim_in * n_w, dim_out))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.weight)

    def get_output(self, G: B, values_list):
        values = torch.cat(values_list, -1) @ self.weight  # [B, |E|, D']
        return batch_like(G, values, skip_masking=True)

    def _0_to_0(self, G: torch.Tensor) -> torch.Tensor:
        # vector -> vector
        return G @ self.weight  # [B, D']

    def _1_to_0(self, G: B) -> torch.Tensor:
        values = G.values  # [B, N, D]
        # set -> vector
        V1 = values.sum(1)  # [B, N, D] -> [B, D]
        if self.normalize:
            n_vec = torch.tensor(G.n_nodes, dtype=V1.dtype, device=V1.device).unsqueeze(-1)  # [B, 1]
            n_vec = n_vec.clone().masked_fill_(n_vec == 0, 1e-5)
            V1 = V1 / n_vec  # [B, D]
        return V1 @ self.weight  # [B, D']

    def _1_to_1(self, G: B) -> B:
        values = G.values  # [B, N, D]
        bsize, n = values.size(0), values.size(1)
        if self.cfg == 'light':
            V1 = values @ self.weight  # [B, N, D']
            return batch_like(G, V1, skip_masking=True)
        assert self.cfg == 'default'
        # set -> set
        V1 = values.clone()  # [B, N, D]
        if self.normalize:
            n_vec = (torch.tensor(G.n_nodes, dtype=V1.dtype, device=V1.device) - 1).view(bsize, 1, 1)  # [B, 1, 1]
            n_vec = n_vec.clone().masked_fill_(n_vec == 0, 1e-5)
            values = values / n_vec  # [B, D]
        M2 = (torch.ones(n, n, device=G.device) - torch.eye(n, device=G.device))  # [N, N]
        V2 = torch.einsum('ji,bid->bjd', M2, values)  # [B, N, D]
        V_list = [V1, V2]
        return self.get_output(G, V_list)

    def _1_to_2(self, G: B) -> B:
        raise NotImplementedError('Sparse set-to-graph is inefficient; use a dense layer')

    def _2_to_0(self, G: B) -> B:
        indices, values = G.indices, G.values  # [B, |E|, 2], [B, |E|, D]
        # graph -> vec
        V1 = d(G).values.sum(1)  # [B, N, D] -> [B, D]
        V2 = nd(G).values.sum(1)  # [B, |E|, D] -> [B, D]
        if self.normalize:
            n_vec = torch.tensor(G.n_nodes, dtype=V1.dtype, device=V1.device).unsqueeze(-1)  # [B, 1]
            V1 = V1 / n_vec.clone().masked_fill_(n_vec == 0, 1e-5)  # [B, D]
            e_vec = (torch.tensor(G.n_edges, dtype=V1.dtype, device=V1.device) - torch.tensor(G.n_nodes, dtype=V1.dtype, device=V1.device)).unsqueeze(-1)  # [B, 1]
            V2 = V2 / e_vec.clone().masked_fill_(e_vec == 0, 1e-5)  # [B, D]
        V_list = [V1, V2]
        V = torch.cat(V_list, -1) @ self.weight  # [B, D']
        return V

    def _2_to_1(self, G: B) -> B:
        indices, values = G.indices, G.values  # [B, |E|, 2], [B, |E|, D]
        bsize = indices.size(0)
        # set -> set
        d_G = d(G)
        diagonal = d_G.values  # [B, N, D], [B, N]
        V1 = diagonal.clone()  # [B, N, D]
        V_list = [V1]
        if self.cfg == 'default':
            n = max(G.n_nodes)
            M_ne = batch_mask(G.node_mask, G.mask)  # [B, N, |E|]
            # get input and output indices
            node_idx = torch.arange(n, device=G.device).view(1, n, 1).expand(bsize, n, 1)  # [B, N, 1]
            row_idx = indices[..., :1]  # [B, |E|, 1]
            col_idx = indices[..., 1:]  # [B, |E|, 1]
            M_l_nc = loop_mask(node_idx, col_idx)
            M_l_nr = loop_mask(node_idx, row_idx)
            M_l_ni = loop_mask(node_idx, indices)
            M_d_nr = diag_mask(node_idx, row_idx)
            M_d_nc = diag_mask(node_idx, col_idx)
            # graph -> set
            M2 = (M_ne & M_l_nc & M_d_nr).float()  # [B, N, |E|]
            V2 = M2.bmm(values)  # [B, N, D]
            M3 = (M_ne & M_l_nr & M_d_nc).float()  # [B, N, |E|]
            V3 = M3.bmm(values)  # [B, N, D]
            if self.normalize:
                V2 = _normalize(V2, M2)
                V3 = _normalize(V3, M3)
            # set -> set
            n_vec = (torch.tensor(G.n_nodes, dtype=V1.dtype, device=V1.device) - 1).view(bsize, 1, 1)
            n_vec = n_vec.clone().masked_fill_(n_vec == 0, 1e-5)
            diagonal_norm = diagonal / n_vec if self.normalize else diagonal  # [B, N, D]
            M4 = (torch.ones(n, n, device=G.device) - torch.eye(n, device=G.device))  # [N, N]
            V4 = torch.einsum('ji,bid->bjd', M4, diagonal_norm)  # [B, N, D]
            # graph -> set
            M5 = (M_ne & M_l_ni).float()  # [B, N, |E|]
            V5 = M5.bmm(values)  # [B, N, D]
            if self.normalize:
                V5 = _normalize(V5, M5)
            V_list += [V2, V3, V4, V5]
            del M_l_nc, M_l_nr, M_l_ni, M_d_nr, M_d_nc
            del M2, M3, M4, M5
        return self.get_output(d_G, V_list)

    def _2_to_2(self, G: B) -> B:
        indices, values = G.indices, G.values  # [B, |E|, 2], [B, |E|, D]
        bsize = indices.size(0)
        diagonal = d(G).values  # [B, N, D]
        V1 = to_diag(indices, diagonal, G.mask, G.node_mask)  # [B, |E|, D]
        V_list = [V1]
        # graph -> graph
        V2 = values
        V3 = do_transpose(values, G.t_indices, G.t_mask)  # [B, |E|, D]
        # set -> graph
        row_idx = indices[..., :1]  # [B, |E|, 1]
        col_idx = indices[..., 1:]  # [B, |E|, 1]
        dim = values.size(-1)
        # broadcast diagonals across columns
        V4 = torch.gather(diagonal, 1, row_idx.clone().clamp_(min=0).expand(bsize, row_idx.size(1), dim))  # [B, |E|, D]
        # broadcast diagonals across rows
        V5 = torch.gather(diagonal, 1, col_idx.clone().clamp_(min=0).expand(bsize, col_idx.size(1), dim))  # [B, |E|, D]
        ND = get_nondiag(indices, torch.cat([V2, V3, V4, V5], -1), G.mask)
        V_list += torch.split(ND, dim, -1)  # [B, |E|, D]
        if self.cfg == 'default':
            n = max(G.n_nodes)
            M_ne = batch_mask(G.node_mask, G.mask)  # [B, N, |E|]
            M_en = batch_mask(G.mask, G.node_mask)  # [B, |E|, N]
            M_ee = batch_mask(G.mask, G.mask)  # [B, |E|, |E|]
            # get input and output indices
            node_idx = torch.arange(n, device=G.device).view(1, n, 1).expand(bsize, n, 1)  # [B, N, 1]
            row_idx = indices[..., :1]  # [B, |E|, 1]
            col_idx = indices[..., 1:]  # [B, |E|, 1]
            M_l_nr = loop_mask(node_idx, row_idx)
            M_l_nc = loop_mask(node_idx, col_idx)
            M_l_ir = loop_mask(indices, row_idx)
            M_l_ic = loop_mask(indices, col_idx)
            M_l_ni = loop_mask(node_idx, indices)
            M_l_in = M_l_ni.transpose(2, 1)
            M_l_ii = loop_mask(indices, indices)
            M_d_nr = diag_mask(node_idx, row_idx)
            M_d_nc = diag_mask(node_idx, col_idx)
            M_d_rr = diag_mask(row_idx, row_idx)
            M_d_cc = diag_mask(col_idx, col_idx)
            M_d_rc = diag_mask(row_idx, col_idx)
            M_d_cr = M_d_rc.transpose(2, 1)
            # graph -> set
            # replicate sum of columns on diagonal
            # replicate sum of rows on diagonal
            M6 = (M_ne & M_l_nc & M_d_nr).float()  # [B, N, |E|]
            M7 = (M_ne & M_l_nr & M_d_nc).float()  # [B, N, |E|]
            V6 = M6.bmm(values)  # [B, N, D]
            V7 = M7.bmm(values)  # [B, N, D]
            if self.normalize:
                V6 = _normalize(V6, M6)
                V7 = _normalize(V7, M7)
            V6 = to_diag(indices, V6, G.mask, G.node_mask)  # [B, |E|, D]
            V7 = to_diag(indices, V7, G.mask, G.node_mask)  # [B, |E|, D]
            # graph -> graph (Oij for i!=j)
            # Oij = sum_{k!=i,j} Akj
            # Oij = sum_{k!=i,j} Aik
            # Oij = sum_{k!=i,j} Ajk
            # Oij = sum_{k!=i,j} Aki
            M8 = (M_ee & M_l_ir & M_d_cc).float()  # [B, |E|, |E|]
            M9 = (M_ee & M_l_ic & M_d_rr).float()  # [B, |E|, |E|]
            M10 = (M_ee & M_l_ic & M_d_cr).float()  # [B, |E|, |E|]
            M11 = (M_ee & M_l_ir & M_d_rc).float()  # [B, |E|, |E|]
            V8 = M8.bmm(values)  # [B, |E|, D]
            V9 = M9.bmm(values)  # [B, |E|, D]
            V10 = M10.bmm(values)  # [B, |E|, D]
            V11 = M11.bmm(values)  # [B, |E|, D]
            if self.normalize:
                V8 = _normalize(V8, M8)
                V9 = _normalize(V9, M9)
                V10 = _normalize(V10, M10)
                V11 = _normalize(V11, M11)
            # set -> set
            n_vec = (torch.tensor(G.n_nodes, dtype=V2.dtype, device=V2.device) - 1).view(bsize, 1, 1)
            diagonal_norm = diagonal / n_vec.clone().masked_fill_(n_vec == 0, 1e-5) if self.normalize else diagonal  # [B, N, D]
            M12 = (torch.ones(n, n, device=G.device) - torch.eye(n, device=G.device))  # [N, N]
            V12 = torch.einsum('ji,bid->bjd', M12, diagonal_norm)  # [B, N, D]
            V12 = to_diag(indices, V12, G.mask, G.node_mask)  # [B, |E|, D]
            # graph -> set
            M13 = (M_ne & M_l_ni).float()  # [B, N, |E|]
            V13 = M13.bmm(values)  # [B, N, D]
            if self.normalize:
                V13 = _normalize(V13, M13)
            V13 = to_diag(indices, V13, G.mask, G.node_mask)  # [B, |E|, D]
            # set -> graph
            # graph -> graph
            M14 = (M_en & M_l_in).float()  # [B, |E|, N]
            M15 = (M_ee & M_l_ii).float()  # [B, |E|, |E|]
            V14 = M14.bmm(diagonal)  # [B, |E|, D]
            V15 = M15.bmm(values)  # [B, |E|, D]
            if self.normalize:
                V14 = _normalize(V14, M14)
                V15 = _normalize(V15, M15)
            V_list += [V6, V7, V8, V9, V10, V11, V12, V13, V14, V15]
            del M6, M7, M8, M9, M10, M11, M12, M13, M14, M15
            del M_l_nc, M_l_nr, M_l_ir, M_l_ic, M_l_in, M_l_ii, M_d_nr, M_d_nc, M_d_cc, M_d_rr, M_d_cr, M_d_rc
        return self.get_output(G, V_list)

    def forward(self, G: Union[B, torch.Tensor]) -> Union[B, torch.Tensor]:
        """Compute the result of A'=LA, where L is defined by combination of equivariant linear basis
        :param G: Batch
        :return: Batch or Tensor([B, D])
        """
        if isinstance(G, B):
            assert G.order == self.ord_in

        if (self.ord_in, self.ord_out) == (0, 0):
            G = self._0_to_0(G)
        elif (self.ord_in, self.ord_out) == (1, 0):
            G = self._1_to_0(G)
        elif (self.ord_in, self.ord_out) == (1, 1):
            G = self._1_to_1(G)
        elif (self.ord_in, self.ord_out) == (1, 2):
            G = self._1_to_2(G)
        elif (self.ord_in, self.ord_out) == (2, 0):
            G = self._2_to_0(G)
        elif (self.ord_in, self.ord_out) == (2, 1):
            G = self._2_to_1(G)
        elif (self.ord_in, self.ord_out) == (2, 2):
            G = self._2_to_2(G)
        else:
            raise NotImplementedError('Currently supports up to second-order only')

        if isinstance(G, B):
            assert G.order == self.ord_out
        return G


class _Bias(nn.Module):
    def __init__(self, ord_out, dim_out):
        super().__init__()
        self.ord_out = ord_out
        if ord_out == 0:
            n_b = 1
        elif ord_out == 1:
            n_b = 1
        elif ord_out == 2:
            n_b = 2
        else:
            raise NotImplementedError('Currently supports up to second-order only')
        self.bias = nn.ParameterList([nn.Parameter(torch.Tensor(dim_out)) for _ in range(n_b)])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        [nn.init.constant_(b, 0.) for b in self.bias]

    def _0(self, G: torch.Tensor) -> torch.Tensor:
        assert len(G.size()) == 2
        bias = self.bias[0][None, :]  # [1, D]
        return G + bias  # [B, D]

    def _1(self, G: B) -> B:
        bias = self.bias[0][None, None, :]  # [1, 1, D]
        return batch_like(G, G.values + bias, skip_masking=True)  # [B, N, D]

    def _2(self, G: B) -> B:
        bsize, dim = G.values.size(0), G.values.size(-1)
        # graph
        b1 = self.bias[0][None, None, :].expand(bsize, G.values.size(1), dim)  # [B, |E|, D]
        bias = get_nondiag(G.indices, b1, G.mask)  # [B, |E|, D]
        # set
        n = max(G.n_nodes)
        b2 = self.bias[1][None, None, :].expand(bsize, n, dim)  # [B, N, D]
        bias = bias + to_diag(G.indices, b2, G.mask, G.node_mask)  # [B, |E|, D]
        return batch_like(G, G.values + bias, skip_masking=True)  # [B, |E|, D]

    def forward(self, G: Union[B, torch.Tensor]) -> Union[B, torch.Tensor]:
        """
        :param G: Batch or Tensor([B, D])
        :return: Batch or Tensor([B, D])
        """
        if isinstance(G, B):
            assert G.order == self.ord_out

        if self.ord_out == 0:
            G = self._0(G)
        elif self.ord_out == 1:
            G = self._1(G)
        elif self.ord_out == 2:
            G = self._2(G)
        else:
            raise NotImplementedError('Currently supports second-order invariance only')

        if isinstance(G, B):
            assert G.order == self.ord_out
        return G


class Linear(nn.Module):
    def __init__(self, ord_in, ord_out, in_dim, out_dim, bias=True, cfg='default', normalize=True):
        super().__init__()
        self.weight = _Weight(ord_in, ord_out, in_dim, out_dim, cfg, normalize)
        if bias:
            self.bias = _Bias(ord_out, out_dim)

    def forward(self, G: Union[B, torch.Tensor]) -> Union[B, torch.Tensor]:
        """
        :param G: Batch or Tensor([B, D])
        :return: Batch or Tensor([B, D'])
        """
        G = self.weight(G)
        if hasattr(self, 'bias'):
            G = self.bias(G)
        if isinstance(G, B):
            G.apply_mask(0)
        return G
