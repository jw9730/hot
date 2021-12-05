import random
import os
import sys
import argparse
import shutil
import json
from pprint import pprint
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Change working directory to project's main directory, and add it to path - for library and config usages
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

# Project dependencies
from dataloaders.delaunay_loader import get_delaunay_loader, get_delaunay_loader_many_sizes
from models import SetToGraph, SetToGraphMLP, SetToGraphGNN, SetToGraphTri, SetToGraphSiam
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
    argparser.add_argument('--res_dir', default='experiments/delaunay_results', help='Results directory')
    argparser.add_argument('--method', default='lin2', type=str, help='method of transitioning from vectors to matrix')
    argparser.add_argument('--baseline', default=None, help='Run on baseline - siam, siam3, mlp or gnn')

    argparser.add_argument('--many_sizes', dest='many_sizes', action='store_true', help='Whether to use n in 20-80 or n=50.')
    argparser.add_argument('--one_size', dest='many_sizes', action='store_false')
    argparser.add_argument('--save', dest='save', action='store_true', help='Whether to save all to disk')
    argparser.add_argument('--no-save', dest='save', action='store_false')
    argparser.set_defaults(save=True, many_sizes=False)

    argparser.add_argument('--scheduler', type=str, default='none', help='Type of learning rate schedule')
    argparser.add_argument('--warmup_epochs', type=int, default=0, help='Learning rate warmup epochs')
    argparser.add_argument('--use_transformer', action='store_true')
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


def update_info(loss, pred, edges, accum_info):
    batch_size = pred.shape[0]
    accum_info['loss'] += loss.item() * batch_size
    pred_edges = pred.ge(0.).float()

    epsilon = 0.00000001
    tp = ((pred_edges == edges) * (pred_edges == 1)).sum(dim=2).sum(dim=1).float()
    tn = ((pred_edges == edges) * (pred_edges == 0)).sum(dim=2).sum(dim=1).float()
    fp = ((pred_edges != edges) * (pred_edges == 1)).sum(dim=2).sum(dim=1).float()
    fn = ((pred_edges != edges) * (pred_edges == 0)).sum(dim=2).sum(dim=1).float()

    accum_info['acc'] += ((tp + tn) / (tp + tn + fp + fn)).sum().item()
    accum_info['precision'] += (tp / (tp + fp + epsilon)).sum().item()
    accum_info['recall'] += (tp / (tp + fn + epsilon)).sum().item()
    accum_info['f1'] += (2 * tp / (2 * tp + fn + fp + epsilon)).sum().item()
    return accum_info


def train_epoch(data, epoch, model, optimizer, scheduler, device):
    model.train()

    # Iterate over batches

    accum_info = {k: 0.0 for k in ['loss', 'acc', 'precision', 'recall', 'f1']}
    for points, edges in tqdm(data):
        # One Train step on the current batch
        points = points.to(device, torch.float)
        edges = edges.to(device, torch.float)

        if isinstance(model, SetToGraphTri):
            pred, loss = model(points, edges)
        else:
            pred = model(points).squeeze(1)  # shape (B,N,N)
            pred = (pred + pred.transpose(1, 2)) / 2

            # calc loss
            loss = F.binary_cross_entropy_with_logits(pred, edges)

        # calc acc, precision, recall
        with torch.no_grad():
            accum_info = update_info(loss, pred, edges, accum_info)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    data_len = len(data.dataset)
    accum_info['loss'] /= data_len
    accum_info['acc'] /= data_len
    accum_info['precision'] /= data_len
    accum_info['recall'] /= data_len
    accum_info['f1'] /= data_len
    print("train epoch %d loss %f acc %f precision %f recall %f f1 %f" % (epoch, accum_info['loss'], accum_info['acc'],
                                                                          accum_info['precision'], accum_info['recall'],
                                                                          accum_info['f1']), flush=True)

    return accum_info


def evaluate(data, epoch, model, device):
    # train epoch
    model.eval()
    accum_info = {k: 0.0 for k in ['loss', 'acc', 'precision', 'recall', 'f1']}

    for points, edges in tqdm(data):
        # One Train step on the current batch
        points = points.to(device, torch.float)
        edges = edges.to(device, torch.float)

        if isinstance(model, SetToGraphTri):
            pred, loss = model(points, edges)
        else:
            pred = model(points).squeeze(1)  # shape (B,N,N)
            pred = (pred + pred.transpose(1, 2)) / 2

            loss = F.binary_cross_entropy_with_logits(pred, edges)

        # calc acc, precision, recall
        accum_info = update_info(loss, pred, edges, accum_info)

    data_len = data.dataset.__len__()
    accum_info['loss'] /= data_len
    accum_info['acc'] /= data_len
    accum_info['precision'] /= data_len
    accum_info['recall'] /= data_len
    accum_info['f1'] /= data_len
    print("validation epoch %d loss %f acc %f precision %f recall %f f1 %f" % (epoch, accum_info['loss'],
                                                                               accum_info['acc'],
                                                                               accum_info['precision'],
                                                                               accum_info['recall'], accum_info['f1']), flush=True)

    return accum_info


def plot_val(df, output_dir, val):
    df.index.name = 'epochs'
    df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    df[['train_'+val, 'val_'+val]].plot(title=val, grid=True)
    plt.savefig(os.path.join(output_dir, val+".pdf"))


def main():
    config = parse_args()
    start_time = datetime.now()
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # uncomment only for CUDA error debugging
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    torch.cuda.set_device(int(config.gpu))
    use_cuda = torch.cuda.is_available()  # returns True
    device = torch.device('cuda' if use_cuda else 'cpu')
    seed = 1728
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True  # can impact performance
    # torch.backends.cudnn.benchmark = False  # can impact performance

    config.exp_name = 'Delaunay'
    pprint(vars(config))
    print(flush=True)

    # Load data
    many_sizes = config.many_sizes
    if many_sizes:
        print('Generating training data, n in 20-80...', flush=True)
        train_data = get_delaunay_loader_many_sizes(config, train=True)
        print('Generating validation data, n in 20-80...', flush=True)
        val_data = get_delaunay_loader_many_sizes(config, train=False)
    else:
        print('Generating training data, n=50...', flush=True)
        train_data = get_delaunay_loader(config, train=True)
        print('Generating validation data, n=50...', flush=True)
        val_data = get_delaunay_loader(config, train=False)

    # Create model instance
    if config.use_transformer:
        # Higher-Order Transformer
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
    else:
        # baselines
        # cfg = dict(agg=torch.mean, normalization='batchnorm', second_bias=False, mlp_with_relu=False)
        if config.baseline == 'mlp':
            maxnodes = 80 if config.many_sizes else 50
            model = SetToGraphMLP([500, 1000, 1000, 1000, 500, 80 ** 2], in_features=2, max_nodes=maxnodes)
        elif config.baseline == 'gnn':
            model = SetToGraphGNN([1000, 1500, 1000], in_features=2, k=5)
        elif config.baseline == 'siam3':
            model = SetToGraphTri([500, 1000, 1500, 1250, 1000, 500, 500, 80], in_features=2)
        elif config.baseline == 'siam':
            cfg = dict(normalization='batchnorm', mlp_with_relu=False)
            model = SetToGraphSiam(2, [700, 700, 700, 1400, 700, 700, 112], hidden_mlp=[1000, 1000], cfg=cfg)
        else:
            cfg = dict(agg=torch.mean)
            model = SetToGraph(in_features=2, out_features=1, set_fn_feats=[500, 500, 500, 1000, 500, 500, 80],
                               method=config.method, hidden_mlp=[1000, 1000], predict_diagonal=True, attention=False, cfg=cfg)

    model = model.to(device)
    print(f'Model: {model}')
    print(f'Num of params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    lr = config.lr
    if config.baseline == 'gnn':
        lr = 1e-4
        print(f'Changed learning rate to 1e-4 for GNN')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    if config.scheduler == 'linear':
        def lambda_rule(it):
            global_epochs = float(it) / len(train_data)
            warmup_epochs = float(config.warmup_epochs)
            lr_w = min(1., global_epochs / warmup_epochs) if config.warmup_epochs > 0 else 1.
            lr_d = 1 - max(0., global_epochs - warmup_epochs) / (config.epochs - warmup_epochs)
            return lr_w * lr_d
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:
        # Fake SCHEDULER
        def lambda_rule(ep):
            return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    # Metrics
    train_loss = np.empty(config.epochs, float)
    train_acc = np.empty(config.epochs, float)
    train_precision = np.empty(config.epochs, float)
    train_recall = np.empty(config.epochs, float)
    train_f1 = np.empty(config.epochs, float)

    val_loss = np.empty(config.epochs, float)
    val_acc = np.empty(config.epochs, float)
    val_precision = np.empty(config.epochs, float)
    val_recall = np.empty(config.epochs, float)
    val_f1 = np.empty(config.epochs, float)

    if not os.path.exists(config.res_dir):
        os.makedirs(config.res_dir)
    exp_dir = f'Delaunay_{start_time:%Y%m%d_%H%M%S}' + config.method
    output_dir = os.path.join(config.res_dir, exp_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Training and evaluation process
    for epoch in range(1, config.epochs + 1):
        train_info = train_epoch(train_data, epoch, model, optimizer, scheduler, device)
        train_loss[epoch - 1], train_acc[epoch - 1], train_precision[epoch - 1], train_recall[epoch - 1], train_f1[epoch - 1] \
            = train_info['loss'], train_info['acc'], train_info['precision'], train_info['recall'], train_info['f1']
        with torch.no_grad():
            val_info = evaluate(val_data, epoch, model, device)
        val_loss[epoch - 1], val_acc[epoch - 1], val_precision[epoch - 1], val_recall[epoch - 1], val_f1[epoch - 1] \
            = val_info['loss'], val_info['acc'], val_info['precision'], val_info['recall'], val_info['f1']

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, os.path.join(output_dir, f'model_epoch{epoch}.pth'))

    # Saving to disk
    if config.save:
        if not os.path.exists(config.res_dir):
            os.makedirs(config.res_dir)
        exp_dir = f'Delaunay_{start_time:%Y%m%d_%H%M%S}' + config.method
        output_dir = os.path.join(config.res_dir, exp_dir)
        os.makedirs(output_dir, exist_ok=True)
        print(f'Saving all to {output_dir}')
        shutil.copyfile(__file__, os.path.join(output_dir, 'code.py'))

        results_dict = {'train_loss': train_loss,
                        'train_acc': train_acc,
                        'train_precision': train_precision,
                        'train_recall': train_recall,
                        'train_f1': train_f1,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'val_precision': val_precision,
                        'val_recall': val_recall,
                        'val_f1': val_f1}
        df = pd.DataFrame(results_dict)
        plot_val(df, output_dir, 'loss')
        plot_val(df, output_dir, 'acc')
        plot_val(df, output_dir, 'precision')
        plot_val(df, output_dir, 'recall')
        plot_val(df, output_dir, 'f1')

        torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))
        if many_sizes:
            torch.save(train_data.dataset.Points, os.path.join(output_dir, 'train_Points.pth'))
            torch.save(train_data.dataset.Edges, os.path.join(output_dir, 'train_Edges.pth'))

            torch.save(val_data.dataset.Points, os.path.join(output_dir, 'val_Points.pth'))
            torch.save(val_data.dataset.Edges, os.path.join(output_dir, 'val_Edges.pth'))
        else:
            torch.save(train_data.dataset.tensors, os.path.join(output_dir, 'train.pth'))
            torch.save(val_data.dataset.tensors, os.path.join(output_dir, 'val.pth'))

        with open(os.path.join(output_dir, 'used_config.json'), 'w') as fp:
            json.dump(vars(config), fp)

    print(f'Total runtime: {str(datetime.now() - start_time).split(".")[0]}')


if __name__ == '__main__':
    main()
