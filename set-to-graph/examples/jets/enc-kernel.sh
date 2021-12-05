#! /bin/bash

python ../../main/main_jets.py \
      --use_transformer \
      --scheduler=linear \
      --warmup_epochs=100 \
      --dim_hidden 128 \
      --dim_qk 128 \
      --dim_v 128 \
      --n_heads 4 \
      --dim_ff 128 \
      --num_hidden 4 \
      --mlp_dim_hidden 256 \
      --mlp_num_hidden 1 \
      --dropout 0.1 \
      --lr 2e-4 \
      -b 512 \
      --test_bs 256 \
      --use_kernel \
      --res_dir experiments/jets/kernel-4l-128dim-128ffn-128qkv-h4-1l-256mlp-b512-lr2e-4 \
