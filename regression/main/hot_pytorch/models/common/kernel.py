"""
Performer (FAVOR+) kernel attention
Modified from https://github.com/lucidrains/performer-pytorch/blob/main/performer_pytorch/performer_pytorch.py
"""
import math
from functools import partial

import torch
import torch.nn as nn


@torch.no_grad()
def orthogonal_matrix_chunk(cols, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block, mode='reduced')
    return q.t()  # [cols, cols]


@torch.no_grad()
def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, device=None):
    """create 2D Gaussian orthogonal matrix"""
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix


class KernelFeatureMap(nn.Module):
    """Randomized feature map for kernelized attention
    phi: Tensor([..., D]) -> Tensor([..., r]); phi(x) > 0
    phi(x) = h(x) / sqrt(m) * [f1(w1^T x), ..., f1(wm^T x), ..., fl(w1^T x), ..., fl(wm^T x)]
    """
    def __init__(self, dim_features, num_features=None, ortho_scaling=0, generalized_attention=False, kernel_fn=nn.ReLU(), feature_redraw_interval=0):
        """
        :param dim_features: D
        :param num_features: m
        :param ortho_scaling: 0 or 1
        :param generalized_attention: whether to use generalized mapping that takes h(x) = sqrt(m) and l=1
        :param kernel_fn: choice of f when using generalized mapping
        :param feature_redraw_interval: interval between random vector redrawing
        """
        super().__init__()
        if num_features is None:
            # m = Theta(d*log(d)), as in Theorem 4
            num_features = int(dim_features * math.log(dim_features))

        self.dim_features = dim_features  # D
        self.num_features = num_features  # m
        self.ortho_scaling = ortho_scaling

        # initialize random vectors (m x D-dim)
        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows=num_features, nb_columns=dim_features, scaling=ortho_scaling)
        projection_matrix = self.create_projection()  # [m, D]
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        self.feature_redraw_interval = feature_redraw_interval
        self.register_buffer('calls_since_last_redraw', torch.tensor(0))

    def softmax_map(self, data, is_query, normalize_data=True, eps=1e-4):
        """as in Lemma 1, l=1, f1=exp, and h(x) = exp(-||x||^2/2)
        -> phi(x) = h(x) / sqrt(m) * exp([w1^T x, ..., wm^T x])
                  = 1    / sqrt(m) * exp([w1^T x - diag_x, ..., wm^T x - diag_x])
        :param data: Tensor([..., D])
        :param is_query: a trick for numerical stability
        :param normalize_data: whether to normalize for scaled dot-product
        :param eps: small positive number for guaranteeing non-negativeness
        :return: Tensor([..., m])
        """
        assert data.size(-1) == self.projection_matrix.size(-1)
        d = data.size(-1)

        data_normalizer = (d ** -0.25) if normalize_data else 1.

        ratio = self.num_features ** -0.5  # 1 / sqrt(m)

        projection = self.projection_matrix.type_as(data)  # [m, D]

        # compute dot product [w1^T x, ..., wm^T x]
        data_dash = torch.einsum('...id,jd->...ij', (data * data_normalizer), projection)  # [..., m]

        # compute diag_x = ||x||^2/2 for later computation of h(x) = exp(-||x||^2/2)
        diag_data = (data ** 2).sum(-1)  # [..., D] -> [...]
        diag_data = (diag_data / 2.0) * (data_normalizer ** 2)  # apply scaling
        diag_data = diag_data.unsqueeze(dim=-1)  # [..., 1]

        # compute phi(x) = 1 / sqrt(m) * exp([w1^T x - diag_x, ..., wm^T x - diag_x])
        if is_query:
            data_dash = ratio * (torch.exp(data_dash - diag_data - torch.max(data_dash, dim=-1, keepdim=True)[0]) + eps)
        else:
            data_dash = ratio * (torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)

        return data_dash.type_as(data)

    def generalized_map(self, data, eps=1e-3, normalize_data=True):
        """phi(x) = h(x) / sqrt(m) * [f(w1^T x), ..., f(wm^T x)]
        :param data: Tensor([..., D])
        :param eps: small positive number for guaranteeing non-negativeness
        :param normalize_data: whether to normalize for scaled dot-product
        :return: Tensor([..., m])
        """
        assert data.size(-1) == self.projection_matrix.size(-1)
        d = data.size(-1)

        data_normalizer = (d ** -0.25) if normalize_data else 1.

        if self.projection_matrix is None:
            return self.kernel_fn(data_normalizer * data) + eps

        projection = self.projection_matrix.type_as(data)  # [m, D]

        # compute dot product [w1^T x, ..., wm^T x]
        data_dash = torch.einsum('...id,jd->...ij', (data * data_normalizer), projection)  # [..., m]

        # compute phi(x) = [f(w1^T x), ..., f(wm^T x)]
        data_prime = self.kernel_fn(data_dash) + eps
        return data_prime.type_as(data)

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def fix_projections_(self):
        self.feature_redraw_interval = None

    def redraw_projections(self):
        if not self.training:
            return

        if (self.feature_redraw_interval is not None) and (self.calls_since_last_redraw >= self.feature_redraw_interval):
            self.redraw_projection_matrix(self.projection_matrix.device)
            self.calls_since_last_redraw.zero_()
            return

        self.calls_since_last_redraw += 1

    def forward(self, x, is_query=False):
        """
        :param x: Tensor([..., D])
        :param is_query: Bool
        :return: Tensor([..., D'])
        """
        # compute (randomized) features of query or key
        # note: this handles scaling in scaled dot-product attention
        if self.generalized_attention:
            return self.generalized_map(x)
        else:
            return self.softmax_map(x, is_query=is_query)
