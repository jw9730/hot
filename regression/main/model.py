from data import get_dataset
from lr import PolynomialDecayLR
import torch
import torch.nn as nn
import pytorch_lightning as pl

from hot_pytorch.models import Encoder, MLP
from hot_pytorch.batch.sparse import Batch, make_batch_concatenated
from hot_pytorch.utils.set import to_masked_batch
from utils.flag import flag_bounded
from utils.laplacian import get_pe


class Model(pl.LightningModule):
    def __init__(self, baseline, n_layers, dim_hidden, dim_qk, dim_v, dim_ff, n_heads,
                 readout_dim_qk, readout_dim_v, readout_n_heads, input_dropout_rate, dropout_rate,
                 weight_decay, dataset_name, warmup_updates, tot_updates, peak_lr, end_lr,
                 flag=False, flag_m=3, flag_step_size=1e-3, flag_mag=1e-3):
        super().__init__()
        self.save_hyperparameters()

        if dataset_name == 'ZINC':
            self.atom_encoder = nn.Embedding(64, dim_hidden, padding_idx=0)
            self.edge_encoder = nn.Embedding(64, dim_hidden, padding_idx=0)
            self.in_degree_encoder = nn.Embedding(64, dim_hidden, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(64, dim_hidden, padding_idx=0)
        else:
            self.atom_encoder = nn.Embedding(512 * 9 + 1, dim_hidden, padding_idx=0)
            self.edge_encoder = nn.Embedding(512 * 3 + 1, dim_hidden, padding_idx=0)
            self.in_degree_encoder = nn.Embedding(512, dim_hidden, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(512, dim_hidden, padding_idx=0)

        self.use_laplacian_pe = False
        if baseline is None:
            ord_hidden = [2] * n_layers
            self.encoder = Encoder(2, 0, ord_hidden, 2 * dim_hidden, dim_hidden, dim_hidden, dim_qk, dim_v, dim_ff, n_heads,
                                   readout_dim_qk, readout_dim_v, readout_n_heads, 'default', 'generalized_kernel', input_dropout_rate, dropout_rate)
        elif baseline == 'mlp':
            ord_hidden = [2] * n_layers
            self.encoder = MLP(2, 0, ord_hidden, 2 * dim_hidden, dim_hidden, dim_hidden, 'relu', dropout_rate)
        elif baseline == 'laplacian':
            self.use_laplacian_pe = True
            self.laplacian_encoder = nn.Linear(512, dim_hidden)
            ord_hidden = [1] * n_layers
            self.encoder = Encoder(1, 0, ord_hidden, 2 * dim_hidden, dim_hidden, dim_hidden, dim_qk, dim_v, dim_ff, n_heads,
                                   readout_dim_qk, readout_dim_v, readout_n_heads, 'default', 'default', input_dropout_rate, dropout_rate)
        else:
            raise RuntimeError('Unknown baseline option')

        if dataset_name == 'PCQM4M-LSC':
            self.out_proj = nn.Linear(dim_hidden, 1)
        else:
            self.downstream_out_proj = nn.Linear(dim_hidden, get_dataset(dataset_name)['num_class'])

        self.evaluator = get_dataset(dataset_name)['evaluator']
        self.metric = get_dataset(dataset_name)['metric']
        self.loss_fn = get_dataset(dataset_name)['loss_fn']
        self.dataset_name = dataset_name

        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.weight_decay = weight_decay

        self.flag = flag
        self.flag_m = flag_m
        self.flag_step_size = flag_step_size
        self.flag_mag = flag_mag
        self.dim_hidden = dim_hidden
        self.automatic_optimization = not self.flag

    def forward(self, batched_data, perturb=None):
        x = batched_data.x
        in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
        edge_index, edge_type = batched_data.edge_index, batched_data.edge_type
        node_num, edge_num = batched_data.node_num, batched_data.edge_num

        edge_feature = self.edge_encoder(edge_type).mean(-2)
        node_feature = self.atom_encoder(x).sum(dim=-2)
        if self.flag and perturb is not None:
            node_feature += perturb
        node_feature = node_feature + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)

        if self.use_laplacian_pe:
            max_node_num = max(node_num)
            edge_index = edge_index.split(edge_num, 1)
            pe = torch.cat([get_pe(i, n, max_node_num) for i, n in zip(edge_index, node_num)])
            node_feature, node_mask = to_masked_batch(node_feature, node_num)
            node_feature = torch.cat([node_feature, self.laplacian_encoder(pe)], dim=-1)
            G = Batch(None, node_feature, node_num, None, node_mask=node_mask)
        else:
            G = make_batch_concatenated(node_feature, edge_index, edge_feature, node_num, edge_num)

        output = self.encoder(G)

        if self.dataset_name == 'PCQM4M-LSC':
            output = self.out_proj(output)
        else:
            output = self.downstream_out_proj(output)
        return output

    def training_step(self, batched_data, batch_idx):
        if self.dataset_name == 'ogbg-molpcba':
            if not self.flag:
                y_hat = self.forward(batched_data).view(-1)
                y_gt = batched_data.y.view(-1).float()
                mask = ~torch.isnan(y_gt)
                loss = self.loss_fn(y_hat[mask], y_gt[mask])
            else:
                y_gt = batched_data.y.view(-1).float()
                mask = ~torch.isnan(y_gt)

                def forward(perturb): return self.forward(batched_data, perturb)
                model_forward = (self, forward)
                n_graph, n_node = batched_data.x.size()[:2]
                perturb_shape = (n_graph, n_node, self.dim_hidden)

                optimizer = self.optimizers()
                optimizer.zero_grad()
                loss, _ = flag_bounded(model_forward, perturb_shape, y_gt[mask], optimizer, batched_data.x.device, self.loss_fn,
                                       m=self.flag_m, step_size=self.flag_step_size, mag=self.flag_mag, mask=mask)
                self.lr_schedulers().step()

        elif self.dataset_name == 'ogbg-molhiv':
            if not self.flag:
                y_hat = self.forward(batched_data).view(-1)
                y_gt = batched_data.y.view(-1).float()
                loss = self.loss_fn(y_hat, y_gt)
            else:
                y_gt = batched_data.y.view(-1).float()
                def forward(perturb): return self.forward(batched_data, perturb)
                model_forward = (self, forward)
                n_graph, n_node = batched_data.x.size()[:2]
                perturb_shape = (n_graph, n_node, self.dim_hidden)

                optimizer = self.optimizers()
                optimizer.zero_grad()
                loss, _ = flag_bounded(model_forward, perturb_shape, y_gt, optimizer, batched_data.x.device, self.loss_fn,
                                       m=self.flag_m, step_size=self.flag_step_size, mag=self.flag_mag)
                self.lr_schedulers().step()
        else:
            y_hat = self.forward(batched_data).view(-1)
            y_gt = batched_data.y.view(-1)
            loss = self.loss_fn(y_hat, y_gt)
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batched_data, batch_idx):
        if self.dataset_name in ['PCQM4M-LSC', 'ZINC']:
            y_pred = self.forward(batched_data).view(-1)
            y_true = batched_data.y.view(-1)
        else:
            y_pred = self.forward(batched_data)
            y_true = batched_data.y
        return {
            'y_pred': y_pred,
            'y_true': y_true,
        }

    def validation_epoch_end(self, outputs):
        y_pred = torch.cat([i['y_pred'] for i in outputs])
        y_true = torch.cat([i['y_true'] for i in outputs])
        if self.dataset_name == 'ogbg-molpcba':
            mask = ~torch.isnan(y_true)
            loss = self.loss_fn(y_pred[mask], y_true[mask])
            self.log('valid_ap', loss, sync_dist=True)
        else:
            input_dict = {"y_true": y_true, "y_pred": y_pred}
            try:
                self.log('valid_' + self.metric, self.evaluator.eval(input_dict)[self.metric], sync_dist=True)
            except:
                pass

    def test_step(self, batched_data, batch_idx):
        if self.dataset_name in ['PCQM4M-LSC', 'ZINC']:
            y_pred = self.forward(batched_data).view(-1)
            y_true = batched_data.y.view(-1)
        else:
            y_pred = self.forward(batched_data)
            y_true = batched_data.y
        return {
            'y_pred': y_pred,
            'y_true': y_true,
            'idx': batched_data.idx,
        }

    def test_epoch_end(self, outputs):
        y_pred = torch.cat([i['y_pred'] for i in outputs])
        y_true = torch.cat([i['y_true'] for i in outputs])
        if self.dataset_name == 'PCQM4M-LSC':
            result = y_pred.cpu().float().numpy()
            idx = torch.cat([i['idx'] for i in outputs])
            torch.save(result, 'y_pred.pt')
            torch.save(idx, 'idx.pt')
            exit(0)
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        self.log('test_' + self.metric, self.evaluator.eval(input_dict)
                 [self.metric], sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay)
        lr_scheduler = {
            'scheduler': PolynomialDecayLR(
                optimizer,
                warmup_updates=self.warmup_updates,
                tot_updates=self.tot_updates,
                lr=self.peak_lr,
                end_lr=self.end_lr,
                power=1.0,
            ),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1,
        }
        return [optimizer], [lr_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Higher-Order Transformer")
        parser.add_argument('--baseline', default=None)
        parser.add_argument('--n_layers', type=int, default=12)
        parser.add_argument('--dim_hidden', type=int, default=256)
        parser.add_argument('--dim_qk', type=int, default=256)
        parser.add_argument('--dim_v', type=int, default=256)
        parser.add_argument('--dim_ff', type=int, default=256)
        parser.add_argument('--n_heads', type=int, default=16)
        parser.add_argument('--readout_dim_qk', type=int, default=256)
        parser.add_argument('--readout_dim_v', type=int, default=256)
        parser.add_argument('--readout_n_heads', type=int, default=16)
        parser.add_argument('--input_dropout_rate', type=float, default=0.)
        parser.add_argument('--dropout_rate', type=float, default=0.)
        parser.add_argument('--weight_decay', type=float, default=0.)
        parser.add_argument('--checkpoint_path', type=str, default='')
        parser.add_argument('--warmup_updates', type=int, default=60000)
        parser.add_argument('--tot_updates', type=int, default=1000000)
        parser.add_argument('--peak_lr', type=float, default=2e-4)
        parser.add_argument('--end_lr', type=float, default=1e-9)
        parser.add_argument('--validate', action='store_true', default=False)
        parser.add_argument('--test', action='store_true', default=False)
        parser.add_argument('--flag', action='store_true')
        parser.add_argument('--flag_m', type=int, default=3)
        parser.add_argument('--flag_step_size', type=float, default=1e-3)
        parser.add_argument('--flag_mag', type=float, default=1e-3)
        parser.add_argument('--profile', action='store_true', default=False)
        return parent_parser
