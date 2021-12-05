#! /bin/bash

python ../../main/main.py \
      --data GPS \
      -f adj \
      --set2graph \
      --n_phi_layers 1 \
      --n_psi_layers 4 \
      --dropout_phi 0 \
      --dropout_psi 0.1 \
      --model_name s2g+
