from typing import Union

import torch
import torch.nn as nn

from ...batch.dense import Batch as B, batch_like, v2d, d
from ...utils.dense import to_diag, get_diag, get_nondiag
from .masksum import mask_tensor, do_masked_sum


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

    def get_output(self, G: B, A_list) -> B:
        A = torch.cat(A_list, -1) @ self.weight  # [B, N, D']
        return batch_like(G, A, skip_masking=True)

    def _0_to_0(self, G: torch.Tensor) -> torch.Tensor:
        # vector -> vector
        return G @ self.weight  # [B, D']

    def _1_to_0(self, G: B) -> torch.Tensor:
        # set -> vector
        A1 = G.A.sum(1)  # [B, N, D] -> [B, D]
        if self.normalize:
            n_vec = torch.tensor(G.n_nodes, device=A1.device, dtype=torch.float).unsqueeze(-1)  # [B, 1]
            n_vec = n_vec.masked_fill_(n_vec == 0, 1e-5)
            A1 = A1 / n_vec  # [B, D]
        return A1 @ self.weight  # [B, D']

    def _1_to_1(self, G: B) -> B:
        A = G.A  # [B, N, D]
        n = A.size(1)
        if self.cfg == 'light':
            A_ = A @ self.weight  # [B, N, D']
            return B(A_, G.n_nodes, G.mask)
        assert self.cfg == 'default'
        M_2 = mask_tensor(2, n, device=A.device)  # [N, N]
        # set -> set
        A1 = A.clone()  # [B, N, D]
        A2 = do_masked_sum(M_2, A, G.mask, normalize=self.normalize)  # [B, N, D]
        A_list = [A1, A2]
        return self.get_output(G, A_list)

    def _1_to_2(self, G: B) -> B:
        A = G.A  # [B, N, D]
        n = A.size(1)
        # set -> set
        A1 = to_diag(A)  # [B, N, N, D]
        A_list = [A1]
        # set -> graph
        A2 = get_nondiag(A.unsqueeze(1).repeat(1, n, 1, 1))  # [B, N, N, D]
        A3 = get_nondiag(A.unsqueeze(2).repeat(1, 1, n, 1))  # [B, N, N, D]
        A_list += [A2, A3]
        if self.cfg == 'default':
            M_2 = mask_tensor(2, n, device=A.device)  # [N, N]
            M_3 = mask_tensor(3, n, device=A.device)  # [N, N, N]
            # set -> set
            A4 = to_diag(do_masked_sum(M_2, A, G.mask, normalize=self.normalize))  # [B, N, D] -> [B, N, N, D]
            # set -> graph
            A5 = do_masked_sum(M_3, A, G.mask, normalize=self.normalize)  # [B, N, N, D]
            A_list += [A4, A5]
        return self.get_output(v2d(G), A_list)

    def _2_to_0(self, G: B) -> torch.Tensor:
        A = G.A  # [B, N, N, D]
        # graph -> vec
        A1 = get_diag(A).sum(1)  # [B, N, D] -> [B, D]
        A2 = get_nondiag(A).sum(1).sum(1)  # [B, N, N, D] -> [B, D]
        if self.normalize:
            n_vec = torch.tensor(G.n_nodes, device=A.device, dtype=torch.float).unsqueeze(-1)  # [B, 1]
            A1 = A1 / n_vec.clone().masked_fill_(n_vec == 0, 1e-5)  # [B, D]
            e_vec = (n_vec.pow(2) - n_vec)  # [B, 1]
            A2 = A2 / e_vec.masked_fill_(e_vec == 0, 1e-5)  # [B, D]
        A_list = [A1, A2]  # [B, D]
        A = torch.cat(A_list, -1) @ self.weight  # [B, D']
        return A

    def _2_to_1(self, G: B) -> B:
        A = G.A  # [B, N, N, D]
        n, dim = A.size(1), A.size(-1)
        d_G = d(G)
        diagonal, node_mask = d_G.A, d_G.mask  # [B, N, D], [B, N]
        if self.cfg == 'light':
            # set -> set
            A1 = diagonal @ self.weight  # [B, N, D] -> [B, N, D']
            return batch_like(d_G, A1)
        M_2 = mask_tensor(2, n, device=A.device)  # [N, N]
        M_3 = mask_tensor(3, n, device=A.device)  # [N, N, N]
        # set -> set
        A1 = diagonal.clone()
        # graph -> set
        A_AT = torch.cat([A, A.transpose(2, 1)], dim=-1)  # [B, N, N, 2D]
        A3_2 = do_masked_sum(M_2, A_AT, node_mask, l=1, normalize=self.normalize, diagonal=(1, 2))
        A3, A2 = A3_2[..., :dim], A3_2[..., dim:]  # [B, N, D], sum of columns/rows
        # set -> set
        A4 = do_masked_sum(M_2, diagonal, node_mask, normalize=self.normalize)  # [B, N, D]
        # graph -> set
        A5 = do_masked_sum(M_3, A, node_mask, normalize=self.normalize)  # [B, N, D]
        A_list = [A1, A2, A3, A4, A5]
        return self.get_output(d_G, A_list)

    def _2_to_2(self, G: B) -> B:
        A = G.A  # [B, N, N, D]
        n, dim = A.size(1), A.size(-1)
        d_G = d(G)
        diagonal, node_mask = d_G.A, d_G.mask  # [B, N, D], [B, N]
        eye = torch.eye(n, device=A.device)
        AT = A.transpose(2, 1)  # [B, N, N, D]
        # set -> set
        A1 = A * eye.view(1, n, n, 1)  # diagonal
        A_list = [A1]
        # graph -> graph
        A2 = get_nondiag(A)  # non-diagonal
        A3 = get_nondiag(AT)  # non-diagonal of transpose
        # set -> graph
        A4 = get_nondiag(diagonal.unsqueeze(2).repeat(1, 1, n, 1))  # [B, N, N, D] non-diagonal of diagonal elements replicated on rows
        A5 = get_nondiag(diagonal.unsqueeze(1).repeat(1, n, 1, 1))  # [B, N, N, D] non-diagonal of diagonal elements replicated on columns
        A_list += [A2, A3, A4, A5]
        if self.cfg == 'default':
            M_2 = mask_tensor(2, n, device=A.device)  # [N, N]
            M_3 = mask_tensor(3, n, device=A.device)  # [N, N, N]
            M_4 = mask_tensor(4, n, device=A.device)  # [N, N, N, N]
            # graph -> set
            A_AT = torch.cat([A, AT], dim=-1)  # [B, N, N, 2D]
            A6_7 = to_diag(do_masked_sum(M_2, A_AT, node_mask, l=1, normalize=self.normalize, diagonal=(1, 2)))
            A6, A7 = A6_7[..., :dim], A6_7[..., dim:]  # [B, N, D], sum of columns/rows replicated on diagonal
            # graph -> graph
            A8_10 = do_masked_sum(M_3, A_AT, node_mask, l=1, normalize=self.normalize, diagonal=(2, 3))
            A8, A10 = A8_10[..., :dim], A8_10[..., dim:]  # [B, N, D]
            A11_9 = do_masked_sum(M_3, A_AT, node_mask, l=1, normalize=self.normalize, diagonal=(1, 3))
            A11, A9 = A11_9[..., :dim], A11_9[..., dim:]  # [B, N, D]
            # set -> set
            A12 = to_diag(do_masked_sum(M_2, diagonal, node_mask, normalize=self.normalize))  # [B, N, D] -> [B, N, N, D]
            # graph -> set
            A13 = to_diag(do_masked_sum(M_3, A, node_mask, normalize=self.normalize))  # [B, N, D] -> [B, N, N, D]
            # set -> graph
            A14 = do_masked_sum(M_3, diagonal, node_mask, normalize=self.normalize)  # [B, N, N, D]
            # graph -> graph
            A15 = do_masked_sum(M_4, A, node_mask, normalize=self.normalize)  # [B, N, N, D]
            A_list += [A6, A7, A8, A9, A10, A11, A12, A13, A14, A15]
        return self.get_output(G, A_list)

    def forward(self, G: Union[B, torch.Tensor]) -> Union[B, torch.Tensor]:
        """Compute the result of A'=LA, where L is defined by combination of equivariant linear basis
        :param G: Batch or Tensor([B, D])
        :return: Batch or Tensor([B, D'])
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

    def _0(self, A: torch.Tensor) -> torch.Tensor:
        assert len(A.size()) == 2
        bias = self.bias[0][None, :]  # [1, D]
        return A + bias  # [B, D]

    def _1(self, G: B) -> B:
        bias = self.bias[0][None, None, :]  # [1, 1, D]
        return batch_like(G, G.A + bias, skip_masking=True)  # [B, N, D]

    def _2(self, G: B) -> B:
        n = G.A.size(1)
        eye = torch.eye(n, device=G.A.device).unsqueeze(-1)  # [N, N, 1]
        # graph
        nondiag = (1 - eye) * self.bias[0][None, None, :]  # [N, N, D]
        bias = nondiag.unsqueeze(0)  # [1, N, N, D]
        # set
        diag = eye * self.bias[1][None, None, :]  # [N, N, D]
        bias += diag.unsqueeze(0)  # [1, N, N, D]
        return batch_like(G, G.A + bias, skip_masking=True)  # [B, N, N, D]

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
    def __init__(self, ord_in, ord_out, dim_in, dim_out, bias=True, cfg='default', normalize=True):
        super().__init__()
        self.weight = _Weight(ord_in, ord_out, dim_in, dim_out, cfg, normalize)
        if bias:
            self.bias = _Bias(ord_out, dim_out)

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
