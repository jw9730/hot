import numpy as np
import time
import torch
import networkx as nx

from hot_pytorch.batch.sparse import make_batch
from hot_pytorch.batch.dense import Batch as D
import hot_pytorch.models


@torch.no_grad()
def get_batched_data(n, bsize, dim, sparse, seed, device='cuda'):
    tic = time.time()
    adj_list = []
    for _ in range(bsize):
        graph = nx.barabasi_albert_graph(n, 5, seed)
        adj = nx.adjacency_matrix(graph).tocoo()
        adj_list.append(adj)
    print(f'Graph init done in \t{time.time() - tic:.3f}sec')

    tic = time.time()

    # initialize features
    assert dim % 2 == 0
    init_list = []
    for adj in adj_list:
        edge_indices = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long, device=device)  # [2, |E|]
        e = edge_indices.size(1)
        node_feat = torch.randn(n, dim // 2, device=device)  # [N, D/2]
        edge_feat = torch.randn(e, dim // 2, device=device)  # [|E|, D/2]
        init_list.append((edge_indices, node_feat, edge_feat))

    if sparse:
        # get sparse batch
        edge_indices, node_features, edge_features = zip(*init_list)
        batch = make_batch(node_features, edge_indices, edge_features)
    else:
        # get dense batch
        A_list = []
        for edge_indices, node_feat, edge_feat in init_list:
            edge_feat = torch.sparse_coo_tensor(edge_indices, edge_feat, size=(n, n, dim // 2)).to_dense()  # [N, N, D/2]
            node_feat = node_feat[None, ...] * torch.eye(n, device=device)[..., None]  # [N, N, D/2]
            A = torch.cat([node_feat, edge_feat], dim=-1)  # [N, N, D]
            A_list.append(A)
        # setup batch
        A = torch.stack(A_list, dim=0)  # [B, N, N, D]
        n_nodes = [n] * bsize
        batch = D(A, n_nodes)

    print(f'Batch init done in \t{time.time() - tic:.3f}sec')
    return batch


def get_peak_mem_and_reset():
    stats = torch.cuda.memory_stats()
    peak_bytes_requirement = stats["allocated_bytes.all.peak"]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    return peak_bytes_requirement / 1024 ** 3  # unit: GB


def measure(model, G):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    out = model(G)  # [B, D]
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()
    forward_t = start.elapsed_time(end) / 1000  # unit: milliseconds

    out = out.sum()

    start.record()
    out.backward()
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()
    backward_t = start.elapsed_time(end) / 1000  # unit: milliseconds

    peak_mem = get_peak_mem_and_reset()
    return forward_t, backward_t, peak_mem


def main_routine(repeat, n, bsize, n_layers, dim, dim_qk, dim_v, n_heads, dim_ff, readout_dim_qk, readout_dim_v, readout_n_heads):
    print(f'\n\nn = {n}')
    result = {}

    print("DL")
    try:
        forward_t, backward_t, peak_mem = [], [], []
        model = hot_pytorch.models.MLP(2, 0, [2] * n_layers, dim, dim, dim, sparse=False).to('cuda')
        for i in range(repeat):
            G = get_batched_data(n, bsize, dim, sparse=False, seed=i)
            ft, bt, pm = measure(model, G)
            forward_t.append(ft)
            backward_t.append(bt)
            peak_mem.append(pm)
        result['dl_forward_t'] = (np.mean(np.array(forward_t)), np.std(np.array(forward_t)))
        result['dl_backward_t'] = (np.mean(np.array(backward_t)), np.std(np.array(backward_t)))
        result['dl_peak_mem'] = (np.mean(np.array(peak_mem)), np.std(np.array(peak_mem)))
    except RuntimeError as e:
        result['dl_forward_t'] = 'OOM'
        result['dl_backward_t'] = 'OOM'
        result['dl_peak_mem'] = 'OOM'
        print(e)

    print("DA")
    try:
        forward_t, backward_t, peak_mem = [], [], []
        model = hot_pytorch.models.Encoder(2, 0, [2] * n_layers, dim, dim, dim, dim_qk, dim_v, dim_ff, n_heads,
                                           readout_dim_qk, readout_dim_v, readout_n_heads, 'default', 'default',
                                           drop_input=0., dropout=0., drop_mu=0., sparse=False).to('cuda')
        for i in range(repeat):
            G = get_batched_data(n, bsize, dim, sparse=False, seed=i)
            ft, bt, pm = measure(model, G)
            forward_t.append(ft)
            backward_t.append(bt)
            peak_mem.append(pm)
        result['da_forward_t'] = (np.mean(np.array(forward_t)), np.std(np.array(forward_t)))
        result['da_backward_t'] = (np.mean(np.array(backward_t)), np.std(np.array(backward_t)))
        result['da_peak_mem'] = (np.mean(np.array(peak_mem)), np.std(np.array(peak_mem)))
    except RuntimeError as e:
        result['da_forward_t'] = 'OOM'
        result['da_backward_t'] = 'OOM'
        result['da_peak_mem'] = 'OOM'

    print("DK")
    try:
        forward_t, backward_t, peak_mem = [], [], []
        model = hot_pytorch.models.Encoder(2, 0, [2] * n_layers, dim, dim, dim, dim_qk, dim_v, dim_ff, n_heads,
                                           readout_dim_qk, readout_dim_v, readout_n_heads, 'default', 'generalized_kernel',
                                           drop_input=0., dropout=0., drop_mu=0., sparse=False).to('cuda')
        model.skip_redraw_projections = True
        for i in range(repeat):
            G = get_batched_data(n, bsize, dim, sparse=False, seed=i)
            ft, bt, pm = measure(model, G)
            forward_t.append(ft)
            backward_t.append(bt)
            peak_mem.append(pm)
        result['dk_forward_t'] = (np.mean(np.array(forward_t)), np.std(np.array(forward_t)))
        result['dk_backward_t'] = (np.mean(np.array(backward_t)), np.std(np.array(backward_t)))
        result['dk_peak_mem'] = (np.mean(np.array(peak_mem)), np.std(np.array(peak_mem)))
    except RuntimeError as e:
        result['dk_forward_t'] = 'OOM'
        result['dk_backward_t'] = 'OOM'
        result['dk_peak_mem'] = 'OOM'
        print(e)

    print("SL")
    try:
        forward_t, backward_t, peak_mem = [], [], []
        model = hot_pytorch.models.MLP(2, 0, [2] * n_layers, dim, dim, dim, sparse=True).to('cuda')
        for i in range(repeat):
            G = get_batched_data(n, bsize, dim, sparse=True, seed=i)
            ft, bt, pm = measure(model, G)
            forward_t.append(ft)
            backward_t.append(bt)
            peak_mem.append(pm)
        result['sl_forward_t'] = (np.mean(np.array(forward_t)), np.std(np.array(forward_t)))
        result['sl_backward_t'] = (np.mean(np.array(backward_t)), np.std(np.array(backward_t)))
        result['sl_peak_mem'] = (np.mean(np.array(peak_mem)), np.std(np.array(peak_mem)))
    except RuntimeError as e:
        result['sl_forward_t'] = 'OOM'
        result['sl_backward_t'] = 'OOM'
        result['sl_peak_mem'] = 'OOM'
        print(e)

    print("SA")
    try:
        forward_t, backward_t, peak_mem = [], [], []
        model = hot_pytorch.models.Encoder(2, 0, [2] * n_layers, dim, dim, dim, dim_qk, dim_v, dim_ff, n_heads,
                                           readout_dim_qk, readout_dim_v, readout_n_heads, 'default', 'default',
                                           drop_input=0., dropout=0., drop_mu=0., sparse=True).to('cuda')
        for i in range(repeat):
            G = get_batched_data(n, bsize, dim, sparse=True, seed=i)
            ft, bt, pm = measure(model, G)
            forward_t.append(ft)
            backward_t.append(bt)
            peak_mem.append(pm)
        result['sa_forward_t'] = (np.mean(np.array(forward_t)), np.std(np.array(forward_t)))
        result['sa_backward_t'] = (np.mean(np.array(backward_t)), np.std(np.array(backward_t)))
        result['sa_peak_mem'] = (np.mean(np.array(peak_mem)), np.std(np.array(peak_mem)))
    except RuntimeError as e:
        result['sa_forward_t'] = 'OOM'
        result['sa_backward_t'] = 'OOM'
        result['sa_peak_mem'] = 'OOM'
        print(e)

    print("SK")
    try:
        forward_t, backward_t, peak_mem = [], [], []
        model = hot_pytorch.models.Encoder(2, 0, [2] * n_layers, dim, dim, dim, dim_qk, dim_v, dim_ff, n_heads,
                                           readout_dim_qk, readout_dim_v, readout_n_heads, 'default', 'generalized_kernel',
                                           drop_input=0., dropout=0., drop_mu=0., sparse=True).to('cuda')
        model.skip_redraw_projections = True
        for i in range(repeat):
            G = get_batched_data(n, bsize, dim, sparse=True, seed=i)
            ft, bt, pm = measure(model, G)
            forward_t.append(ft)
            backward_t.append(bt)
            peak_mem.append(pm)
        result['sk_forward_t'] = (np.mean(np.array(forward_t)), np.std(np.array(forward_t)))
        result['sk_backward_t'] = (np.mean(np.array(backward_t)), np.std(np.array(backward_t)))
        result['sk_peak_mem'] = (np.mean(np.array(peak_mem)), np.std(np.array(peak_mem)))
    except RuntimeError as e:
        result['sk_forward_t'] = 'OOM'
        result['sk_backward_t'] = 'OOM'
        result['sk_peak_mem'] = 'OOM'
        print(e)

    return result


def main():
    repeat = 10
    bsize = 1
    n_layers = 4
    dim = 32
    dim_qk = 32
    dim_v = 32
    n_heads = 4
    dim_ff = 32
    readout_dim_qk = 32
    readout_dim_v = 32
    readout_n_heads = 4
    result = {}
    n_list = list((2 ** np.linspace(5, 18, 27, endpoint=True)).astype(int) // 5)  # for log-scale plot
    for n in n_list:
        start = time.time()
        n_result = main_routine(repeat, n, bsize, n_layers, dim, dim_qk, dim_v, n_heads, dim_ff, readout_dim_qk, readout_dim_v, readout_n_heads)
        print(f"{n}: done after {(time.time() - start):.2f} sec")
        print(f"result_{n} = {n_result}")
        result[n] = n_result
        if n_result['sk_forward_t'] == 'OOM':
            break
    print(result)


if __name__ == '__main__':
    main()
