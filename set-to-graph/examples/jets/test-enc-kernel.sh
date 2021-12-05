#! /bin/bash
# change checkpoint before use

python ../../main/test_jets.py \
      --use_transformer \
      --dim_hidden 128 \
      --dim_qk 128 \
      --dim_v 128 \
      --n_heads 4 \
      --dim_ff 128 \
      --num_hidden 4 \
      --mlp_dim_hidden 256 \
      --mlp_num_hidden 1 \
      --dropout 0.1 \
      -b 512 \
      --test_bs 512 \
      --use_kernel \
      --res_dir experiments/jets/... \
      --checkpoint experiments/jets/.../model.pth
