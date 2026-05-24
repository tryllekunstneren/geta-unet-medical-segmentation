#!/bin/bash
#BSUB -J encdec_small_geta      # Job name
#BSUB -q gpuv100                # Queue
#BSUB -n 4                      # CPU cores
#BSUB -R "rusage[mem=8GB]"      # Memory per core
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00                  # Wall time
#BSUB -o /dtu/blackhole/15/187541/geta_outputs/encdec_small_geta_s30_%J.out
#BSUB -e /dtu/blackhole/15/187541/geta_outputs/encdec_small_geta_s30_%J.err

# ---- Activate conda environment ----
source ~/.bashrc
conda activate geta

# ---- Navigate to repo root ----
cd /zhome/88/a/187541/Desktop/bach/geta

# ---- Set paths ----
DATA_DIR=/dtu/blackhole/15/187541/geta_data
OUT_DIR=/dtu/blackhole/15/187541/geta_outputs/encdec_small_geta_s30

mkdir -p $OUT_DIR

# ---- Run GETA (plain encoder-decoder, 30% sparsity) ----
python train_unet_geta.py \
    --model encdec_small \
    --data_root $DATA_DIR \
    --img_size 256 \
    --batch_size 16 \
    --num_workers 4 \
    --epochs 150 \
    --lr 1e-3 \
    --lr_quant 1e-3 \
    --weight_decay 1e-4 \
    --variant adam \
    --sparsity 0.3 \
    --projection_start_epoch 0 \
    --projection_periods 5 \
    --projection_epochs 15 \
    --pruning_start_epoch 15 \
    --pruning_periods 5 \
    --pruning_epochs 50 \
    --bit_reduction 2 \
    --min_bit_wt 4 \
    --max_bit_wt 16 \
    --out_dir $OUT_DIR \
    --seed 0
