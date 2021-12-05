# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_edge_idx_unsqueeze(x, padlen):
    xlen = x.size(1)
    if xlen < padlen:
        new_x = x.new_zeros([2, padlen], dtype=x.dtype).fill_(0)
        new_x[:, :xlen] = x
        new_x[:, xlen:] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


class Batch:
    def __init__(self, idx, edge_index, edge_type, in_degree, out_degree, x, y, node_num, edge_num):
        super(Batch, self).__init__()
        self.idx = idx
        self.in_degree, self.out_degree = in_degree, out_degree
        self.x, self.y = x, y
        self.edge_index, self.edge_type = edge_index, edge_type
        self.node_num, self.edge_num = node_num, edge_num

    def to(self, device):
        self.idx = self.idx.to(device)
        self.in_degree, self.out_degree = self.in_degree.to(device), self.out_degree.to(device)
        self.x, self.y = self.x.to(device), self.y.to(device)
        self.edge_index, self.edge_type = self.edge_index.to(device), self.edge_type.to(device)
        return self

    def __len__(self):
        return len(self.node_num)


def collator(items, max_node=512):
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [(item.idx, item.edge_index, item.edge_type, item.in_degree, item.out_degree, item.x, item.y) for item in items]
    idxs, edge_indices, edge_types, in_degrees, out_degrees, xs, ys = zip(*items)
    node_num = [i.size(0) for i in xs]
    edge_num = [i.size(1) for i in edge_indices]
    y = torch.cat(ys)
    x_ = torch.cat(xs) + 1  # [N, D]
    edge_index_ = torch.cat(edge_indices, dim=1)  # [2, |E|], has no self-loops
    edge_type_ = torch.cat(edge_types)  # [|E|, D]
    in_degree_ = torch.cat(in_degrees) + 1  # [N,]
    out_degree_ = torch.cat(out_degrees) + 1  # [N,]
    return Batch(idx=torch.LongTensor(idxs),
                 edge_index=edge_index_, edge_type=edge_type_,
                 in_degree=in_degree_, out_degree=out_degree_,
                 x=x_, y=y,
                 node_num=node_num, edge_num=edge_num)
