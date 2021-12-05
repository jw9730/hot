import torch
import torch.nn as nn
from .layers import Attention


class DeepSet(nn.Module):
    def __init__(self, in_features, feats, attention, cfg=None):
        """
        DeepSets implementation
        :param in_features: input's number of features
        :param feats: list of features for each deepsets layer
        :param attention: True/False to use attention
        :param cfg: configurations of second_bias and normalization method
        """
        super(DeepSet, self).__init__()
        if cfg is None:
            cfg = {}

        layers = []
        normalization = cfg.get('normalization', 'fro')
        second_bias = cfg.get('second_bias', True)

        layers.append(DeepSetLayer(in_features, feats[0], attention, normalization, second_bias))
        for i in range(1, len(feats)):
            layers.append(nn.ReLU())
            layers.append(DeepSetLayer(feats[i-1], feats[i], attention, normalization, second_bias))

        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)


class DeepSetLayer(nn.Module):
    def __init__(self, in_features, out_features, attention, normalization, second_bias):
        """
        DeepSets single layer
        :param in_features: input's number of features
        :param out_features: output's number of features
        :param attention: Whether to use attention
        :param normalization: normalization method - 'fro' or 'batchnorm'
        :param second_bias: use a bias in second conv1d layer
        """
        super(DeepSetLayer, self).__init__()

        self.attention = None
        if attention:
            self.attention = Attention(in_features)
        self.layer1 = nn.Conv1d(in_features, out_features, 1)
        self.layer2 = nn.Conv1d(in_features, out_features, 1, bias=second_bias)

        self.normalization = normalization
        if normalization == 'batchnorm':
            self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        # x.shape = (B,C,N)

        # attention
        if self.attention:
            x_T = x.transpose(2, 1)  # B,C,N -> B,N,C
            x = self.layer1(x) + self.layer2(self.attention(x_T).transpose(1, 2))
        else:
            x = self.layer1(x) + self.layer2(x - x.mean(dim=2, keepdim=True))

        # normalization
        if self.normalization == 'batchnorm':
            x = self.bn(x)
        else:
            x = x / torch.norm(x, p='fro', dim=1, keepdim=True)  # BxCxN / Bx1xN

        return x
