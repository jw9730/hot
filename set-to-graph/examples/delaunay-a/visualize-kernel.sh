#! /bin/bash
# change checkpoint_path and baseline_checkpoint_path before use

python ../../main/visualize_delaunay.py \
      --dim_hidden 256 \
      --dim_qk 256 \
      --dim_v 256 \
      --n_heads 4 \
      --dim_ff 256 \
      --num_hidden 5 \
      --mlp_dim_hidden 256 \
      --mlp_num_hidden 2 \
      --dropout 0.1 \
      --use_kernel \
      -b 1 \
      --checkpoint_path experiments/delaunay-a/.../model.pth \
      --baseline_checkpoint_path experiments/delaunay-a/.../model.pth
