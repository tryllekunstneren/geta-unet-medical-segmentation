#!/bin/bash
#BSUB -J unet_baseline          # Job name
#BSUB -q gpuv100                # Queue
#BSUB -n 4                      # CPU cores
#BSUB -R "rusage[mem=8GB]"      # Memory per core
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 6:00                   # Wall time
#BSUB -o /dtu/blackhole/15/187541/geta_outputs/unet_baseline_%J.out
#BSUB -e /dtu/blackhole/15/187541/geta_outputs/unet_baseline_%J.err

# ---- Activate conda environment ----
source ~/.bashrc
conda activate geta

# ---- Navigate to repo root ----
cd /zhome/88/a/187541/Desktop/bach/geta

# ---- Set paths ----
DATA_DIR=/dtu/blackhole/15/187541/geta_data
OUT_DIR=/dtu/blackhole/15/187541/geta_outputs/unet_baseline

mkdir -p $OUT_DIR

# ---- Run baseline training ----
python train_unet_baseline.py \
    --data_root $DATA_DIR \
    --img_size 256 \
    --batch_size 16 \
    --num_workers 4 \
    --epochs 100 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --out_dir $OUT_DIR \
    --seed 0
