# GETA for U-Net Medical Image Segmentation

Code for my BSc thesis at the Technical University of Denmark (DTU),
*Automatic Joint Structured Pruning and Quantization for Efficient
Neural Network Training and Compression*.

This project applies the GETA framework (Qu et al., 2025) to U-Net
and encoder-decoder architectures for medical image segmentation,
across three datasets: LGG Brain MRI, BraTS 2020, and ISIC 2018.

## Overview

GETA performs joint structured pruning and quantization in a single
training run. This repository contains the model definitions,
dataloaders, training scripts, and HPC submission scripts used to
evaluate GETA on medical segmentation tasks, along with a sequential
prune-then-quantize baseline for comparison.

## Dependencies

This project builds on the original GETA / Only-Train-Once framework.
To run the code you need to install it separately:

1. Clone the GETA framework from the original repository:
   (https://github.com/microsoft/GETA)
2. Set up the conda environment from `environment.yml`:
   ```
   conda env create -f environment.yml
   conda activate geta
   ```

## Repository structure

```
src/
  models/        U-Net variants and the EncDec variant
  data/          Dataloaders for LGG, BraTS, and ISIC
  training/      Training scripts (GETA, sequential, baselines)
scripts/         HPC job submission and data download scripts
docs/            Notes, including the stability-fix patch
```

## Datasets

The datasets are not included in this repository due to size and
licensing. They can be obtained from:

- **LGG Brain MRI**: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
- **BraTS 2020**: https://www.med.upenn.edu/cbica/brats2020/
- **ISIC 2018 (Task 1)**: https://challenge.isic-archive.com/data/#2018

The download helper scripts in `scripts/` assist with fetching the
ISIC and BraTS data.

## Running experiments

Training is launched through the submission scripts in `scripts/`.
For example, to run GETA on U-Net at 30% sparsity:

```
bsub < scripts/run_unet_geta_s30.sh
```

The main training entry points are:

- `src/training/train_unet_geta.py` — GETA joint compression
- `src/training/train_sequential.py` — sequential prune-then-quantize baseline
- `src/training/train_unet_baseline.py` — uncompressed baselines
- `src/training/train_resnet20_cifar10.py` — ResNet20/CIFAR10 validation

Key arguments for the GETA script include `--dataset`, `--model`,
`--sparsity`, `--min_bit_wt`, and `--max_bit_wt`.

## Models

The U-Net variants share the same depth (four encoder/decoder
levels) and block structure, differing only in base channel count:

- `unet_tiny` (16, 32, 64, 128)
- `unet_small` (32, 64, 128, 256)
- `unet_standard` (64, 128, 256, 512)
- `unet_large` (96, 192, 384, 768)

The `encdec_small` variant has the same structure as `unet_small`
but removes the skip connections, used to study their effect on
compression.

## Stability fix

During the experiments we found a numerical instability in the QASSO
optimiser that, in the reference implementation, can stop training
with an `UnboundLocalError`. A description and a minimal patch are
provided in `docs/geta_patch.md`.

## Acknowledgements

This work builds directly on the GETA framework by Qu et al. (2025).
All credit for the framework itself goes to the original authors.

## Author

Sofus Carstens (s224959), DTU, 2026.
