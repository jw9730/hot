# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from argparse import ArgumentParser
from pprint import pprint
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.profiler import AdvancedProfiler

from model import Model
from data import GraphDataModule, get_dataset


def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Model.add_model_specific_args(parser)
    parser = GraphDataModule.add_argparse_args(parser)
    args = parser.parse_args()
    args.max_steps = args.tot_updates + 1
    if not args.test and not args.validate:
        print(args)
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    dm = GraphDataModule.from_argparse_args(args)

    # ------------
    # model
    # ------------
    if args.checkpoint_path != '':
        model = Model.load_from_checkpoint(
            args.checkpoint_path,
            strict=False,
            baseline=args.baseline,
            n_layers=args.n_layers,
            dim_hidden=args.dim_hidden,
            dim_qk=args.dim_qk,
            dim_v=args.dim_v,
            dim_ff=args.dim_ff,
            n_heads=args.n_heads,
            readout_dim_qk=args.readout_dim_qk,
            readout_dim_v=args.readout_dim_v,
            readout_n_heads=args.readout_n_heads,
            input_dropout_rate=args.input_dropout_rate,
            dropout_rate=args.dropout_rate,
            weight_decay=args.weight_decay,
            dataset_name=dm.dataset_name,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            flag=args.flag,
            flag_m=args.flag_m,
            flag_step_size=args.flag_step_size
        )
    else:
        model = Model(
            baseline=args.baseline,
            n_layers=args.n_layers,
            dim_hidden=args.dim_hidden,
            dim_qk=args.dim_qk,
            dim_v=args.dim_v,
            dim_ff=args.dim_ff,
            n_heads=args.n_heads,
            readout_dim_qk=args.readout_dim_qk,
            readout_dim_v=args.readout_dim_v,
            readout_n_heads=args.readout_n_heads,
            input_dropout_rate=args.input_dropout_rate,
            dropout_rate=args.dropout_rate,
            weight_decay=args.weight_decay,
            dataset_name=dm.dataset_name,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            flag=args.flag,
            flag_m=args.flag_m,
            flag_step_size=args.flag_step_size
        )
    if not args.test and not args.validate:
        print(model)
    print('total params:', sum(p.numel() for p in model.parameters()))

    # ------------
    # training
    # ------------
    metric = 'valid_' + get_dataset(dm.dataset_name)['metric']
    dirpath = args.default_root_dir + f'/lightning_logs/checkpoints'
    checkpoint_callback = ModelCheckpoint(
        monitor=metric,
        dirpath=dirpath,
        filename=dm.dataset_name + '-{epoch:03d}-{' + metric + ':.4f}',
        save_top_k=100,
        mode=get_dataset(dm.dataset_name)['metric_mode'],
        save_last=True,
    )
    if not args.test and not args.validate and os.path.exists(dirpath + '/last.ckpt'):
        args.resume_from_checkpoint = dirpath + '/last.ckpt'
        print('args.resume_from_checkpoint', args.resume_from_checkpoint)

    if args.profile:
        trainer = pl.Trainer.from_argparse_args(args, plugins=DDPPlugin(find_unused_parameters=args.baseline == 'laplacian'),
                                                profiler=AdvancedProfiler(filename='perf.txt'))
    else:
        trainer = pl.Trainer.from_argparse_args(args, plugins=DDPPlugin(find_unused_parameters=args.baseline == 'laplacian'))

    trainer.callbacks.append(checkpoint_callback)
    trainer.callbacks.append(LearningRateMonitor(logging_interval='step'))

    if args.test:
        result = trainer.test(model, datamodule=dm)
        pprint(result)
    elif args.validate:
        result = trainer.validate(model, datamodule=dm)
        pprint(result)
    else:
        trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    cli_main()
