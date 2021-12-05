"""https://github.com/graphdeeplearning/graphtransformer/blob/c9cd49368eed4507f9ae92a137d90a7a9d7efc3a/data/SBMs.py#L145"""
import torch
import numpy as np
from scipy import sparse as sp


def get_pe(indices: torch.LongTensor, n_node, pad_size, pos_enc_dim=512):
    i = indices.detach().cpu().numpy()
    A = sp.coo_matrix((np.ones(i.shape[1]), (i[0], i[1])), shape=(n_node, n_node))
    N = sp.diags(np.asarray(A.sum(1)).squeeze(1).clip(1) ** -0.5, dtype=float)
    L = sp.eye(n_node) - N * A * N
    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    EigVec = torch.from_numpy(EigVec[:, :pos_enc_dim + 1]).float().to(indices.device)  # [N, D]
    pe = torch.zeros(pad_size, pos_enc_dim).to(indices.device)
    pe[:n_node, :min(pos_enc_dim + 1, n_node)] = EigVec
    return pe.unsqueeze(0)
