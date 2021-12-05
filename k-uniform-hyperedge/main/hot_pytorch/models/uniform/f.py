from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class Nonlinear(nn.Module):
    def __init__(self, f: str, skip_masking=False):
        super().__init__()
        self.f = {'relu': F.relu, 'leakyrelu': F.leaky_relu, 'gelu': F.gelu, 'tanh': F.tanh}[f]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)


class Apply(nn.Module):
    def __init__(self, f: Callable[[torch.Tensor], torch.Tensor], skip_masking=False):
        super().__init__()
        self.f = f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)


class Add(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return x1 + x2
