import torch
import torch.nn as nn


class SetToGraphTri(nn.Module):
    def __init__(self, params, in_features=2):
        """
        Triplets model.
        """
        super().__init__()
        last = in_features
        layers = []
        for p in params:
            layers.append(nn.Linear(last, p))
            layers.append(nn.ReLU())
            last = p
        layers = layers[:-1]
        self.model = nn.Sequential(*layers)

    def forward(self, x, gt):
        device = x.device
        x = self.model(x)  # shape (B,N,C_out)

        B, N, C = x.shape
        loss = torch.tensor(0., requires_grad=True, device=device)

        dists = (x.unsqueeze(1) - x.unsqueeze(2)).pow(2).sum(3)  # shape (B,N,N)

        tri = torch.randint(0, N, (200, 3), device=device)
        tri = tri[tri[:, 0] != tri[:, 1]]
        tri = tri[tri[:, 0] != tri[:, 2]]
        tri = tri[tri[:, 1] != tri[:, 2]]

        if gt is not None:
            for i in range(B):
                if gt[i].unique().numel() == 1:
                    continue  # only one label, cant learn from it
                tri_i = tri[(gt[i, tri[:, 0], tri[:, 1]].bool()) & (~gt[i, tri[:, 0], tri[:, 2]].bool())]
                tri_i = tri_i.unique(dim=0)
                if len(tri_i) == 0:
                    continue
                anch, pos, neg = tri_i.t()
                loss = loss + torch.clamp_min(dists[i, anch, pos]-dists[i, anch, neg]+2, 0.).mean()

        #graphs = (dists + dists.transpose(1, 2)) / 2  # to make symmetric
        graphs = dists.le(1.).float() # adj matrix - 1 as threshold
        graphs = graphs - 0.5  # the main script uses 0 as threshold
        return graphs, loss
