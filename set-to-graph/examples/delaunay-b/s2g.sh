#! /bin/bash
# S2G for comparative visualization

python ../../main/main_delaunay.py \
      --scheduler=linear \
      --warmup_epochs=10 \
      --lr 2e-4 \
      --many_sizes \
      --res_dir experiments/delaunay-b/s2g \
