import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GraphConv


class SetToGraphGNN(torch.nn.Module):
    def __init__(self, params, in_features=2, k=10):
        super().__init__()
        self.k = k
        last = in_features
        self.convs = nn.ModuleList()
        for p in params:
            self.convs.append(GraphConv(last, p))
            last = p

    def forward(self, x):
        b, n, c = x.shape

        k = self.k
        if k >= n:
            k = n - 1

        # k nearest neighbors
        nbors = torch.topk(torch.norm(x.unsqueeze(1) - x.unsqueeze(2), dim=3), k + 1, largest=False)[1][:, :,
                1:]  # shape b,n,k
        src = torch.arange(n, device=x.device).reshape(1, n, 1).repeat(b, 1, 1).repeat(1, 1, k).flatten()  # shape b*n*k
        edge_index = torch.stack([src, nbors.flatten()])  # shape 2, b*n*k
        batch = torch.arange(b, device=x.device).reshape(b, 1).repeat(1, k * n).flatten()  # shape b*n*k
        edge_index = edge_index + (batch.view(1, -1) * n)  # batched graphes
        x = x.view(b * n, c)

        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)

        x = self.convs[-1](x, edge_index)  # shape b*n,c_new
        x = x.view(b, n, -1)  # shape b,n,c_new
        graphs = x @ x.transpose(1, 2)  # outer product, shape b,n,n
        graphs = graphs.unsqueeze(1)  # shape b,1,n,n

        return graphs
