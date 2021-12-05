import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PsiSuffix(nn.Module):
    def __init__(self, features, predict_diagonal):
        super().__init__()
        layers = []
        for i in range(len(features) - 2):
            layers.append(DiagOffdiagMLP(features[i], features[i + 1], predict_diagonal))
            layers.append(nn.ReLU())
        layers.append(DiagOffdiagMLP(features[-2], features[-1], predict_diagonal))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DiagOffdiagMLP(nn.Module):
    def __init__(self, in_features, out_features, seperate_diag):
        super(DiagOffdiagMLP, self).__init__()

        self.seperate_diag = seperate_diag
        self.conv_offdiag = nn.Conv2d(in_features, out_features, 1)
        if self.seperate_diag:
            self.conv_diag = nn.Conv1d(in_features, out_features, 1)

    def forward(self, x):
        # Assume x.shape == (B, C, N, N)
        if self.seperate_diag:
            return self.conv_offdiag(x) + (self.conv_diag(x.diagonal(dim1=2, dim2=3))).diag_embed(dim1=2, dim2=3)
        return self.conv_offdiag(x)


class Attention(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        small_in_features = max(math.floor(in_features/10), 1)
        self.d_k = small_in_features

        self.query = nn.Sequential(
            nn.Linear(in_features, small_in_features),
            nn.Tanh(),
        )
        self.key = nn.Linear(in_features, small_in_features)

    def forward(self, inp):
        # inp.shape should be (B,N,C)
        q = self.query(inp)  # (B,N,C/10)
        k = self.key(inp)  # B,N,C/10

        x = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)  # B,N,N

        x = x.transpose(1, 2)  # (B,N,N)
        x = x.softmax(dim=2)  # over rows
        x = torch.matmul(x, inp)  # (B, N, C)
        return x
