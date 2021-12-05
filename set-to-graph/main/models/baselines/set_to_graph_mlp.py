import torch
import torch.nn as nn


class SetToGraphMLP(torch.nn.Module):
    def __init__(self, params, in_features=2, max_nodes=80):
        super().__init__()
        self.max_nodes = max_nodes
        last = in_features * max_nodes
        layers = []
        for p in params:
            layers.append(nn.Linear(last, p))
            layers.append(nn.ReLU())
            last = p
        layers = layers[:-1]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        B, N, C = x.shape
        new_x = torch.zeros((B, self.max_nodes, C), device=x.device)
        new_x[:, :N] = x

        new_x = new_x.view(B, self.max_nodes * C).contiguous()
        graph = self.model(new_x).view(B, 80, 80).contiguous()[:, :N, :N]  # B,N,N
        graph = (graph + graph.transpose(1, 2)) / 2  # symmetric
        graph = graph.unsqueeze(1)  # shape b,1,n,n

        return graph
