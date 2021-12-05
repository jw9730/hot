import torch
import torch.nn as nn

from hot_pytorch.models.extension import Encoder, MLP


class FFN(nn.Module):
    def __init__(self, dims, dropout=None, use_bias=True, residual=False, layer_norm=False):
        super(FFN, self).__init__()
        self.w_stack = []
        self.dims = dims
        for i in range(len(dims) - 1):
            self.w_stack.append(nn.Conv1d(dims[i], dims[i + 1], 1, bias=use_bias))
            self.add_module("PWF_Conv%d" % i, self.w_stack[-1])
        self.layer_norm = nn.LayerNorm(dims[-1])
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.residual = residual
        self.layer_norm_flag = layer_norm

    def forward(self, x):
        output = x.unsqueeze(1).transpose(1, 2)
        for i in range(len(self.w_stack) - 1):
            output = self.w_stack[i](output)
            output = torch.relu(output)
            if self.dropout is not None:
                output = self.dropout(output)
        output = self.w_stack[-1](output)
        output = output.transpose(1, 2)
        if self.dims[0] == self.dims[-1]:
            # residual
            if self.residual:
                output += x
            if self.layer_norm_flag:
                output = self.layer_norm(output)
        return output.squeeze(1)


class EncoderS2G(nn.Module):
    def __init__(self, dim_in, dim_out, set_fn_feats, dim_qk, dim_v, dim_ff, n_heads, dropout_phi, dropout_psi, hidden_mlp, simple_mlp):
        super().__init__()
        # layer 1 + 2: set-to-set + set-to-edge
        dim_hidden = set_fn_feats[0] if len(set_fn_feats) > 0 else dim_in
        for h in set_fn_feats:
            assert h == dim_hidden, 'transformer only allows constant hidden dimensions'
        ord_hidden = [1] * len(set_fn_feats)
        self.enc = Encoder(1, 3, ord_hidden, dim_in, dim_hidden, dim_hidden, dim_qk, dim_v, dim_ff, n_heads, 0, 0, 0,
                           'default', 'generalized_kernel', 0., dropout_phi)
        # layer 3: edge-to-edge
        dims = [dim_in] + hidden_mlp + [dim_out]
        if simple_mlp:
            psi_layers = list()
            for idx, pair in enumerate(zip(dims[:-1], dims[1:])):
                dim1, dim2 = pair
                psi_layers.append(nn.Linear(dim1, dim2))
                if idx < len(dims) - 2:
                    psi_layers.append(nn.ReLU())
                    psi_layers.append(nn.Dropout(dropout_psi))
            self.suffix = nn.Sequential(*psi_layers)
        else:
            self.suffix = FFN(dims, dropout=dropout_psi, use_bias=True, residual=True, layer_norm=True)

    def forward(self, x, indices):
        try:
            return self.suffix(self.enc(x, indices))
        except AssertionError as e:
            print(f'Warning: This should not happen normally\n{e}')
            indices = torch.LongTensor([list(range(indices.size(1)))] * indices.size(0)).to(indices.device)
            return self.suffix(self.enc(x, indices))


class MLPS2G(nn.Module):
    def __init__(self, dim_in, dim_out, set_fn_feats, dropout_phi, dropout_psi, hidden_mlp):
        super().__init__()
        # layer 1 + 2: set-to-set + set-to-edge
        ord_hidden = [1] * len(set_fn_feats)
        self.enc = MLP(1, 3, ord_hidden, dim_in, hidden_mlp[0], set_fn_feats, 'relu', dropout_phi)
        # layer 3: edge-to-edge
        psi_layers = list()
        psi_layers.append(nn.ReLU())
        psi_layers.append(nn.Dropout(dropout_psi))
        dims = hidden_mlp + [dim_out]
        for idx, pair in enumerate(zip(dims[:-1], dims[1:])):
            dim1, dim2 = pair
            psi_layers.append(nn.Linear(dim1, dim2, bias=True))
            if idx < len(dims) - 2:
                psi_layers.append(nn.ReLU())
                psi_layers.append(nn.Dropout(dropout_psi))
        self.suffix = nn.Sequential(*psi_layers)

    def forward(self, x, indices):
        try:
            return self.suffix(self.enc(x, indices))
        except AssertionError as e:
            print(f'Warning: This should not happen normally\n{e}')
            indices = torch.LongTensor([list(range(indices.size(1)))] * indices.size(0)).to(indices.device)
            return self.suffix(self.enc(x, indices))
