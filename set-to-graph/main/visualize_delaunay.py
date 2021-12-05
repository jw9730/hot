import random
import os
import sys
import argparse
import torch
from tqdm import tqdm
from pyvis.network import Network
import networkx as nx

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

# Project dependencies
from dataloaders.delaunay_vis_loader import get_delaunay_loader, get_delaunay_loader_many_sizes
from models import SetToGraph
from models import EncoderS2G


def parse_args():
    """
    Define and retrieve command line arguements
    :return: argparser instance
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-g', '--gpu', default='0', help='The gpu to run on')
    argparser.add_argument('-e', '--epochs', default=100, type=int, help='The number of epochs to run')
    argparser.add_argument('-l', '--lr', default=0.001, type=float, help='The learning rate')
    argparser.add_argument('-b', '--bs', default=64, type=int, help='Batch size to use')
    argparser.add_argument('--n_examples_test', default=5000, type=int, help='number of test examples')
    argparser.add_argument('--n_examples', default=50000, type=int, help='number of training examples')
    argparser.add_argument('--method', default='lin2', type=str, help='method of transitioning from vectors to matrix')
    argparser.add_argument('--baseline', default=None, help='Run on baseline - siam, siam3, mlp or gnn')

    argparser.add_argument('--many_sizes', dest='many_sizes', action='store_true', help='Whether to use n in 20-80 or n=50.')
    argparser.add_argument('--one_size', dest='many_sizes', action='store_false')
    argparser.add_argument('--save', dest='save', action='store_true', help='Whether to save all to disk')
    argparser.add_argument('--no-save', dest='save', action='store_false')
    argparser.set_defaults(save=True, many_sizes=False)

    argparser.add_argument('--checkpoint_path', type=str, required=True)
    argparser.add_argument('--baseline_checkpoint_path', type=str, required=True)
    argparser.add_argument('--dim_hidden', type=int, default=256)
    argparser.add_argument('--dim_qk', type=int, default=256)
    argparser.add_argument('--dim_v', type=int, default=256)
    argparser.add_argument('--n_heads', type=int, default=4)
    argparser.add_argument('--dim_ff', type=int, default=256)
    argparser.add_argument('--num_hidden', type=int, default=4)
    argparser.add_argument('--mlp_dim_hidden', type=int, default=256)
    argparser.add_argument('--mlp_num_hidden', type=int, default=2)
    argparser.add_argument('--drop_input', type=float, default=0.)
    argparser.add_argument('--dropout', type=float, default=0.)
    argparser.add_argument('--use_kernel', action='store_true')

    args = argparser.parse_args()
    return args


def main():
    config = parse_args()

    if not os.path.exists('visualize_delaunay'):
        os.makedirs('visualize_delaunay')

    # Generate Test Data
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    test_data = []
    node_num = 10
    many_size = True
    if many_size:
        for node_num in [20, 30, 40, 50, 60, 70, 80]:
            test_data += get_delaunay_loader_many_sizes(config, False, node_num)
    else:
        test_data = get_delaunay_loader(config, train=False)

    # Manually input features & Load model using checkpoints
    model = EncoderS2G(dim_in=2,
                       dim_out=1,
                       set_fn_feats=[config.dim_hidden] * config.num_hidden,
                       dim_qk=config.dim_qk,
                       dim_v=config.dim_v,
                       dim_ff=config.dim_ff,
                       n_heads=config.n_heads,
                       use_kernel=config.use_kernel,
                       drop_input=config.drop_input,
                       dropout=config.dropout,
                       hidden_mlp=[config.mlp_dim_hidden] * config.mlp_num_hidden,
                       predict_diagonal=True)

    checkpoint = torch.load(config.checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    # Put data into model
    print(f"generating graph data")

    # use_cuda = torch.cuda.is_available()  # returns True

    model = model.to(device)

    idx = 1
    print(len(test_data))
    with torch.no_grad():
        for points, edges in tqdm(test_data):
            # For single graph for now
            points = points.to(device, torch.float)
            edges = edges.to(device, torch.float)

            # Drawing Ground Truth
            node_pos = points.squeeze(0).cpu().numpy()
            gt = edges.squeeze(0).cpu().numpy()
            G_gt = nx.from_numpy_matrix(gt)

            for v in G_gt.nodes:
                G_gt.nodes[v]['pos'] = node_pos[v].tolist()
            for u, v in G_gt.edges:
                G_gt[u][v]['width'] = 4

            pv_G_gt = Network()
            pv_G_gt.width = '2000px'
            pv_G_gt.from_nx(G_gt, default_node_size=3)

            for v in pv_G_gt.nodes:
                v['label'] = ''
                v['color'] = 'black'
                v.update({'physics': False})
                v['size'] = 10
                v['x'] = node_pos[v['id']].tolist()[0] * 1000
                v['y'] = node_pos[v['id']].tolist()[1] * 1000

            pv_G_gt.show(f'visualize_delaunay/delaunay_triangulation_gt_{idx}.html')

            # Drawing Prediction
            pred = model(points).squeeze(1)  # shape (B,N,N)
            pred = pred.squeeze(0)  # shape (N,N)
            pred = (pred + pred.transpose(0, 1)) / 2
            pred = (pred > 0).float()
            np_pred = pred.cpu().numpy()
            G = nx.from_numpy_matrix(np_pred)

            for v in G.nodes:
                G.nodes[v]['pos'] = node_pos[v].tolist()

            for u, v in G.edges:
                G[u][v]['width'] = 4

            # Add Red (Inaccurate) edges
            for u, v in G.edges:
                if not G_gt.has_edge(u, v):
                    G[u][v]['color'] = 'red'
                    G[u][v]['width'] = 7
            # Add Blue (Missing) edges
            for u, v in G_gt.edges:
                if not G.has_edge(u, v):
                    G.add_edge(u, v, color='blue')
                    G[u][v]['width'] = 7

            pv_G = Network()
            pv_G.width = '2000px'
            pv_G.from_nx(G, default_node_size=3)
            for v in pv_G.nodes:
                v['label'] = ''
                v['color'] = 'black'
                v.update({'physics': False})
                v['size'] = 10
                v['x'] = node_pos[v['id']].tolist()[0] * 1000
                v['y'] = node_pos[v['id']].tolist()[1] * 1000

            pv_G.show(f'visualize_delaunay/delaunay_triangulation_transformer_{idx}.html')
            idx = idx + 1
    model.to('cpu')

    cfg = dict(agg=torch.mean)
    model = SetToGraph(in_features=2,
                       out_features=1,
                       set_fn_feats=[500, 500, 500, 1000, 500, 500, 80],
                       method=config.method,
                       hidden_mlp=[1000, 1000],
                       predict_diagonal=True,
                       attention=False,
                       cfg=cfg)

    checkpoint = torch.load(config.baseline_checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    model = model.to(device)
    idx = 1
    with torch.no_grad():
        for points, edges in tqdm(test_data):
            points = points.to(device, torch.float)
            edges = edges.to(device, torch.float)
            node_pos = points.squeeze(0).cpu().numpy()

            gt = edges.squeeze(0).cpu().numpy()
            G_gt = nx.from_numpy_matrix(gt)

            # Drawing Prediction
            pred = model(points).squeeze(1)  # shape (B,N,N)
            pred = pred.squeeze(0)  # shape (N,N)
            pred = (pred + pred.transpose(0, 1)) / 2
            pred = (pred > 0).float()
            np_pred = pred.cpu().numpy()
            G = nx.from_numpy_matrix(np_pred)

            for v in G.nodes:
                G.nodes[v]['pos'] = node_pos[v].tolist()

            for u, v in G.edges:
                G[u][v]['width'] = 4

            # Add Red (Inaccurate) edges
            for u, v in G.edges:
                if not G_gt.has_edge(u, v):
                    G[u][v]['color'] = 'red'
                    G[u][v]['width'] = 7
            # Add Blue (Missing) edges
            for u, v in G_gt.edges:
                if not G.has_edge(u, v):
                    G.add_edge(u, v, color='blue')
                    G[u][v]['width'] = 7

            pv_G = Network()
            pv_G.width = '2000px'
            pv_G.from_nx(G, default_node_size=3)

            for v in pv_G.nodes:
                v['label'] = ''
                v['color'] = 'black'
                v.update({'physics': False})
                v['size'] = 10
                v['x'] = node_pos[v['id']].tolist()[0] * 1000
                v['y'] = node_pos[v['id']].tolist()[1] * 1000
            pv_G.show(f'visualize_delaunay/delaunay_triangulation_s2g_{idx}.html')
            idx = idx + 1


if __name__ == '__main__':
    main()
