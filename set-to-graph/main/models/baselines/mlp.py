import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_features, feats, cfg=None):
        """
        Element wise MLP implementation
        :param in_features: input's number of features
        :param feats: list of features for each linear layer
        :param cfg: configurations of to end with relu and normalization method
        """
        super().__init__()

        if cfg is None:
            cfg = {}
        self.end_with_relu = cfg.get('mlp_with_relu', True)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv1d(in_features, feats[0], 1))
        for i in range(1, len(feats)):
            self.layers.append(nn.Conv1d(feats[i-1], feats[i], 1))

        self.normalization = cfg.get('normalization', 'fro')
        if self.normalization == 'batchnorm':
            self.bns = nn.ModuleList([nn.BatchNorm1d(feat) for feat in feats])

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.normalization == 'batchnorm':
                x = self.bns[i](x)
            else:
                x = x / torch.norm(x, p='fro', dim=1, keepdim=True)  # BxCxN / Bx1xN
            x = F.relu(x)

        x = self.layers[-1](x)
        if self.normalization == 'batchnorm':
            x = self.bns[-1](x)
        else:
            x = x / torch.norm(x, p='fro', dim=1, keepdim=True)  # BxCxN / Bx1xN
        if self.end_with_relu:
            x = F.relu(x)

        return x
