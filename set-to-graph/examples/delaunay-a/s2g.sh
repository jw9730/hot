#! /bin/bash
# S2G for comparative visualization

python ../../main/main_delaunay.py \
      --scheduler=linear \
      --warmup_epochs=10 \
      --lr 2e-4 \
      --res_dir experiments/delaunay-a/s2g \
