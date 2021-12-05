#! /bin/bash

python ../../main/main_delaunay.py \
      --use_transformer \
      --scheduler=linear \
      --warmup_epochs=10 \
      --dim_hidden 256 \
      --dim_qk 256 \
      --dim_v 256 \
      --n_heads 4 \
      --dim_ff 256 \
      --num_hidden 5 \
      --mlp_dim_hidden 256 \
      --mlp_num_hidden 2 \
      --dropout 0.1 \
      --lr 2e-4 \
      --many_sizes \
      -b 16 \
      --res_dir experiments/delaunay-b/softmax-5l-256dim-256ffn-256qkv-h4-2l-256mlp-b16-lr2e-4 \
