import argparse
import logging
import os
import warnings

import torch
import torch.nn as nn
from tqdm import tqdm

from unet import unet_tiny, unet_small, unet_standard, unet_large
from encoder_decoder import encdec_small, encdec_standard
from brain_mri_dataset import get_brain_mri_loaders
from isic_dataset import get_isic_loaders
from brats_dataset import get_brats_loaders


def build_model(model_name, in_channels=3, out_channels=1):
    models = {
        "unet_tiny": unet_tiny,
        "unet_small": unet_small,
        "unet_standard": unet_standard,
        "unet_large": unet_large,
        "encdec_small": encdec_small,
        "encdec_standard": encdec_standard,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(models)}")
    return models[model_name](in_channels=in_channels, out_channels=out_channels)

warnings.filterwarnings("ignore")
logger = logging.getLogger("unet_baseline")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def dice_coefficient(pred, target, smooth=1e-6):
    """Compute Dice coefficient (2 * intersection / union)."""
    pred = torch.sigmoid(pred)
    pred_binary = (pred > 0.5).float()
    intersection = (pred_binary * target).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()


def iou_score(pred, target, smooth=1e-6):
    """Compute IoU / Jaccard index."""
    pred = torch.sigmoid(pred)
    pred_binary = (pred > 0.5).float()
    intersection = (pred_binary * target).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------
class DiceBCELoss(nn.Module):
    """Combined Binary Cross Entropy + Dice Loss for segmentation."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(1, 2, 3))
        union = pred_sigmoid.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = dice_loss.mean()

        return bce_loss + dice_loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n_batches = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, masks)

        total_loss += loss.item()
        total_dice += dice_coefficient(outputs, masks).item()
        total_iou += iou_score(outputs, masks).item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "dice": total_dice / n_batches,
        "iou": total_iou / n_batches,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main(cfg):
    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Logging
    log_file = os.path.join(out_dir, f"{cfg.model}_baseline.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger.info("=" * 60)
    logger.info(f"Baseline  —  {cfg.model} on Brain MRI (no compression)")
    logger.info("=" * 60)
    for k, v in vars(cfg).items():
        logger.info(f"  {k:30s}: {v}")
    logger.info("=" * 60)

    # Seed
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed(cfg.seed)
    logger.info(f"Device: {device}")

    # Data
    if cfg.dataset == "lgg":
        train_loader, val_loader = get_brain_mri_loaders(
            data_root=os.path.join(cfg.data_root, "lgg-mri-segmentation"),
            img_size=cfg.img_size,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            val_split=0.2,
            seed=cfg.seed,
        )
    elif cfg.dataset == "isic":
        train_loader, val_loader = get_isic_loaders(
            data_root=os.path.join(cfg.data_root, "isic2018"),
            img_size=cfg.img_size,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
    else:  # brats
        train_loader, val_loader = get_brats_loaders(
            data_root=os.path.join(cfg.data_root, "brats2020", "BraTS2020_training_data"),
            img_size=cfg.img_size,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            val_split=0.2,
            seed=cfg.seed,
        )

    # Model
    model = build_model(cfg.model, in_channels=3, out_channels=1).to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"{cfg.model} parameters: {num_params:.2f} M")

    # Loss, optimizer, scheduler
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-6
    )

    # Training
    best_dice = 0.0
    best_epoch = 0

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        n_batches = 0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}", leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += dice_coefficient(outputs, masks).item()
            n_batches += 1

        scheduler.step()

        train_loss = running_loss / n_batches
        train_dice = running_dice / n_batches

        # Validation
        val_metrics = evaluate(model, val_loader, criterion, device)

        logger.info(
            f"Epoch {epoch:>3d}/{cfg.epochs} | "
            f"train_loss {train_loss:.4f} | train_dice {train_dice:.4f} | "
            f"val_loss {val_metrics['loss']:.4f} | "
            f"val_dice {val_metrics['dice']:.4f} | "
            f"val_iou {val_metrics['iou']:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(out_dir, f"{cfg.model}_baseline_best.pt"))

    logger.info("=" * 60)
    logger.info("BASELINE RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Model              : {cfg.model}")
    logger.info(f"  Best val Dice      : {best_dice:.4f}")
    logger.info(f"  Best val IoU       : {val_metrics['iou']:.4f}")
    logger.info(f"  Best epoch         : {best_epoch}")
    logger.info(f"  Parameters         : {num_params:.2f} M")
    logger.info("=" * 60)
    logger.info("Done!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def get_config():
    p = argparse.ArgumentParser(description="Baseline U-Net ISIC 2018")

    p.add_argument("--model", default="unet_small",
                   choices=["unet_tiny", "unet_small", "unet_standard", "unet_large",
                            "encdec_small", "encdec_standard"],
                   help="Model architecture to train")
    p.add_argument("--dataset", default="lgg", choices=["lgg", "isic", "brats"],
                   help="Dataset: lgg (Brain MRI), isic (skin lesion), brats (BraTS 2020)")
    p.add_argument("--data_root", required=True,
                   help="Root data directory; loader appends lgg-mri-segmentation/ or isic2018/")
    p.add_argument("--img_size", type=int, default=256,
                   help="Resize images to this size")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="./outputs/unet_baseline")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    return p.parse_args()


if __name__ == "__main__":
    main(get_config())
