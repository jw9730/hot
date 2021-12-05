[ -z "${exp_name}" ] && exp_name="laplacian-100k-8l-256dim-256ffn-16qkv-h16-256read-h16-b1024-lr1e-4"
[ -z "${seed}" ] && seed="1"
[ -z "${arch}" ] && arch="--tot_updates 100000 --warmup_updates 5000 --peak_lr 1e-4 --baseline laplacian --n_layers 8 --dim_hidden 256 --dim_ff 256 --dim_qk 256 --dim_v 256 --n_heads 16 --readout_dim_qk 256 --readout_dim_v 256 --readout_n_heads 16 --weight_decay 1e-5 --input_dropout_rate 0.1 --dropout_rate 0.1"
[ -z "${batch_size}" ] && batch_size="128"

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "exp_name: ${exp_name}"
echo "arch: ${arch}"
echo "seed: ${seed}"
echo "batch_size: ${batch_size}"
echo "==============================================================================="

default_root_dir="../../exps/pcq/$exp_name/$seed"
mkdir -p $default_root_dir
n_gpu=$(nvidia-smi -L | wc -l)

python ../../main/entry.py --num_workers 8 --seed $seed --batch_size $batch_size \
      --dataset_name PCQM4M-LSC \
      --gpus $n_gpu --accelerator ddp --precision 32 --gradient_clip_val 5.0 \
      $arch \
      --default_root_dir $default_root_dir \
