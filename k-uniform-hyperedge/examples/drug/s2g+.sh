#! /bin/bash

python ../../main/main.py \
      --data drug \
      -f adj \
      --set2graph \
      --n_phi_layers 3 \
      --n_psi_layers 2 \
      --dropout_phi 0 \
      --dropout_psi 0.1 \
      --model_name s2g+
