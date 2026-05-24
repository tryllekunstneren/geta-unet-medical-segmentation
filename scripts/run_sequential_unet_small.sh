#!/bin/bash
#BSUB -J sequential_unet_s30    # Job name
#BSUB -q gpuv100                # Queue
#BSUB -n 4                      # CPU cores
#BSUB -R "rusage[mem=8GB]"      # Memory per core
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 16:00                  # Wall time (100 + 50 epochs = 150 total)
#BSUB -o /dtu/blackhole/15/187541/geta_outputs/sequential_unet_small_s30_%J.out
#BSUB -e /dtu/blackhole/15/187541/geta_outputs/sequential_unet_small_s30_%J.err

# ---- Activate conda environment ----
source ~/.bashrc
conda activate geta

# ---- Navigate to repo root ----
cd /zhome/88/a/187541/Desktop/bach/geta

# ---- Set paths ----
DATA_DIR=/dtu/blackhole/15/187541/geta_data
OUT_DIR=/dtu/blackhole/15/187541/geta_outputs/sequential_unet_small_s30

mkdir -p $OUT_DIR

# ---- Run sequential baseline (prune first, then quantize) ----
# Phase 1: 100 epochs HESSO structured pruning (no quantization)
# Phase 2:  50 epochs quantization-aware fine-tuning of pruned model
# Total: 150 epochs — same training budget as GETA for fair comparison
python train_sequential.py \
    --model unet_small \
    --data_root $DATA_DIR \
    --img_size 256 \
    --batch_size 16 \
    --num_workers 4 \
    --seed 0 \
    --out_dir $OUT_DIR \
    --variant adam \
    --lr 1e-3 \
    --lr_finetune 1e-4 \
    --lr_quant 1e-3 \
    --weight_decay 1e-4 \
    --sparsity 0.3 \
    --phase1_epochs 100 \
    --pruning_start_epoch 10 \
    --pruning_periods 5 \
    --pruning_epochs 70 \
    --phase2_epochs 50
