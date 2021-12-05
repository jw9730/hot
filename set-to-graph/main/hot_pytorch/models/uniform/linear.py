import torch
import torch.nn as nn


class _Weight(nn.Module):
    def __init__(self, ord_in, ord_out, dim_in, dim_out, cfg='default', normalize=False):
        super().__init__()
        assert cfg in ('default', 'light')
        self.ord_in = ord_in
        self.ord_out = ord_out
        self.dim_in = dim_in
        self.cfg = cfg
        self.normalize = normalize
        if ord_in == 1:
            n_w = {'default': ord_out + 1, 'light': ord_out}[cfg]
        else:
            raise NotImplementedError('This extension is only for 1->k-uniform')
        self.weight = nn.Parameter(torch.Tensor(dim_in * n_w, dim_out))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.weight)

    def _1_to_1(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg == 'light':
            return x @ self.weight  # [N, D']
        assert self.cfg == 'default'
        # set -> set
        x1 = x.clone()  # [N, D]
        x2 = torch.mean(x, dim=0, keepdim=True) if self.normalize else torch.sum(x, dim=0, keepdim=True)  # [1, D]
        return x1 @ self.weight[:-self.dim_in] + x2 @ self.weight[-self.dim_in:]  # [N, D']

    def _1_to_k(self, x: torch.Tensor, indices: torch.LongTensor) -> torch.Tensor:
        x_list = list()
        for idx_vec in indices.unbind(-1):
            idx_vec = idx_vec[:, None].expand(idx_vec.size(0), x.size(1))  # [B, D]
            x_list.append(torch.gather(x, dim=0, index=idx_vec, sparse_grad=True))  # [B, D]
        x_nonpool = torch.cat(x_list, -1)
        if self.cfg == 'light':
            return x_nonpool @ self.weight  # [B, D']
        assert self.cfg == 'default'
        x_pool = torch.mean(x, dim=0, keepdim=True) if self.normalize else torch.sum(x, dim=0, keepdim=True)  # [1, D]
        return x_nonpool @ self.weight[:-self.dim_in] + x_pool @ self.weight[-self.dim_in:]  # [B, D']

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


class Linear(nn.Module):
    def __init__(self, ord_in, ord_out, in_dim, out_dim, bias=True, cfg='default', normalize=True):
        super().__init__()
        self.weight = _Weight(ord_in, ord_out, in_dim, out_dim, cfg, normalize)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
            self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.bias, 0.)

    def forward(self, x: torch.Tensor, indices: torch.LongTensor = None) -> torch.Tensor:
        """
        :param x: Tensor([N, D])
        :param indices: Tensor([B, k]) or None
        :return: Tensor([B, D']) or Tensor([N, D'])
        """
        x = self.weight(x, indices)
        if hasattr(self, 'bias'):
            x = x + self.bias[None, :]
        return x
