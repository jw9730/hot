import torch
import torch.nn as nn
import numpy as np

from .mlp import MLP


class SetPartitionTri(nn.Module):
    def __init__(self, in_features, mlp_features):
        """
        SetPartitionTri model.
        """
        super().__init__()
        cfg = dict(mlp_with_relu=False)
        self.mlp = MLP(in_features=in_features, feats=mlp_features, cfg=cfg)
        self.tensor_1 = torch.tensor(1., device='cuda')

    def forward(self, x, labels, margin=2.):
        device = x.device

        x = x.transpose(2, 1)  # from BxNxC to BxCxN
        u = self.mlp(x)  # Bx(out_features)xN
        u = u.transpose(2, 1)  # shape BxNx(out_features)

        B, N, C = u.shape
        loss = torch.tensor(0., requires_grad=True, device=device)

        dists = (u.unsqueeze(1) - u.unsqueeze(2)).pow(2).sum(3)  # shape (B,N,N)

        tri = torch.randint(0, N, (200, 3), device=device)
        tri = tri[tri[:, 0] != tri[:, 1]]

        if labels is not None:
            for i in range(B):
                if labels[i].max().item() == 0:
                    continue  # only one cluster, cant learn from it
                tri_i = tri[(labels[i, tri[:, 0]] == labels[i, tri[:, 1]]) & (labels[i, tri[:, 1]] != labels[i, tri[:, 2]])]
                tri_i = tri_i.unique(dim=0)
                if len(tri_i) == 0:
                    continue

                anch, pos, neg = tri_i.t()
                loss = loss + torch.clamp_min(dists[i, anch, pos]-dists[i, anch, neg]+2, 0.).mean()

        # deployment - infer chosen clusters:
        with torch.no_grad():
            pred_matrices = (dists + dists.transpose(1, 2)) / 2  # to make symmetric
            pred_matrices = pred_matrices.le(1.).float()  # adj matrix - 1 as threshold
            pred_matrices[:, np.arange(N), np.arange(N)] = self.tensor_1  # each node is always connected to itself
            ones_now = pred_matrices.sum()
            ones_before = ones_now - 1
            while ones_now != ones_before:  # get connected components - each node connected to all in its component
                ones_before = ones_now
                pred_matrices = torch.matmul(pred_matrices, pred_matrices)
                pred_matrices = pred_matrices.bool().float()  # remain as 0-1 matrices
                ones_now = pred_matrices.sum()

            clusters = -1 * torch.ones((B, N), device=device)
            for i in range(N):
                clusters = torch.where(pred_matrices[:, i] == 1, i * self.tensor_1, clusters)

        return clusters.long(), loss

    def generate_triplets(self, labels, n_triplets):
        tries = 0
        triplets = []
        labels = labels.cpu().numpy()
        while tries < 25 and len(triplets) < n_triplets:
            tries += 1
            idx = np.random.randint(0, labels.shape[0])
            idx_matches = np.where(labels == labels[idx])[0]
            idx_no_matches = np.where(labels != labels[idx])[0]
            if len(idx_matches) > 1 and len(idx_no_matches) > 0:
                idx_a, idx_p = np.random.choice(idx_matches, 2, replace=False)
                idx_n = np.random.choice(idx_no_matches, 1)[0]
                triplets.append([idx_a, idx_p, idx_n])
        return np.array(triplets)
