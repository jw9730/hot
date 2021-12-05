import torch
import torch.nn as nn


class MuDropout(nn.Module):
    """In-place mu-dropout"""
    def __init__(self, p=0.):
        super().__init__()
        self.p = p  # drop probability

    def forward(self, x_list):
        if not self.training or self.p == 0.:
            return x_list
        else:
            n_mu = len(x_list)
            bsize = len(x_list[0])
            n_dim = len(x_list[0].size())
            rand_size = [n_mu, bsize] + [1] * (n_dim - 1)
            keep_mask = (torch.rand(rand_size, device=x_list[0].device) > self.p).float()
            return [x * m / (1 - self.p) for x, m in zip(x_list, keep_mask)]
