import torch
import torch.nn as nn


class SetPartitionMLP(torch.nn.Module): 
    def __init__(self, params, in_features=10): 
        super().__init__() 
        assert params[-1] == 15**2

        last = in_features * 15 
        layers = [] 
        for p in params: 
            layers.append(nn.Linear(last, p)) 
            layers.append(nn.ReLU())
            last = p 
        layers = layers[:-1]
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        B, N, C = x.shape
        new_x = torch.zeros((B, 15, C), device=x.device)
        new_x[:, :N] = x

        new_x = new_x.view(B, 15*C).contiguous()
        edge_vals = self.model(new_x).view(B, 15, 15).contiguous()[:, :N, :N].unsqueeze(1)  # B,1,N,N

        return edge_vals
