#! /bin/bash

python ../../main/main.py \
      --data MovieLens \
      -f adj \
      --set2graph \
      --use_transformer \
      --n_phi_layers 3 \
      --n_psi_layers 2 \
      --dropout_phi 0 \
      --dropout_psi 0.1 \
      --model_name enc
