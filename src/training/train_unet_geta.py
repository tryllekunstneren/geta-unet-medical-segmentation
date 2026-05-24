import argparse
import json
import logging
import math
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
from only_train_once import OTO
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode

warnings.filterwarnings("ignore")
logger = logging.getLogger("unet_geta")


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Dataset loader selector
# ---------------------------------------------------------------------------
def get_loaders(cfg):
    if cfg.dataset == "lgg":
        return get_brain_mri_loaders(
            data_root=os.path.join(cfg.data_root, "lgg-mri-segmentation"),
            img_size=cfg.img_size,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            val_split=0.2,
            seed=cfg.seed,
        )
    elif cfg.dataset == "isic":
        return get_isic_loaders(
            data_root=os.path.join(cfg.data_root, "isic2018"),
            img_size=cfg.img_size,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
    else:  # brats
        return get_brats_loaders(
            data_root=os.path.join(cfg.data_root, "brats2020", "BraTS2020_training_data"),
            img_size=cfg.img_size,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            val_split=0.2,
            seed=cfg.seed,
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def dice_coefficient(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred_binary = (pred > 0.5).float()
    intersection = (pred_binary * target).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2.0 * intersection + smooth) / (union + smooth)).mean()


def iou_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred_binary = (pred > 0.5).float()
    intersection = (pred_binary * target).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + smooth) / (union + smooth)).mean()


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(1, 2, 3))
        union = pred_sigmoid.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        return bce_loss + dice_loss.mean()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice, total_iou, n = 0.0, 0.0, 0.0, 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        outputs = model(images)
        total_loss += criterion(outputs, masks).item()
        total_dice += dice_coefficient(outputs, masks).item()
        total_iou += iou_score(outputs, masks).item()
        n += 1
    return {"loss": total_loss / n, "dice": total_dice / n, "iou": total_iou / n}


# ---------------------------------------------------------------------------
# Quantization parameter inspection
# ---------------------------------------------------------------------------
def get_quant_param_dict(model):
    param_dict = {}
    for name, param in model.named_parameters():
        if any(k in name for k in ("d_quant", "t_quant", "q_m")):
            layer_name = ".".join(name.split(".")[:-1])
            param_name = name.split(".")[-1]
            param_dict.setdefault(layer_name, {})[param_name] = param.item()
    return param_dict


def get_bitwidth_dict(param_dict):
    bit_dict = {}
    for key, vals in param_dict.items():
        bit_dict[key] = {}
        d = vals["d_quant_wt"]
        q = abs(vals["q_m_wt"])
        t = vals.get("t_quant_wt", 1.0)
        bit_dict[key]["weight"] = math.log2(math.exp(t * math.log(q)) / abs(d) + 1) + 1
        if "d_quant_act" in vals:
            da = vals["d_quant_act"]
            qa = abs(vals["q_m_act"])
            ta = vals.get("t_quant_act", 1.0)
            bit_dict[key]["activation"] = math.log2(math.exp(ta * math.log(qa)) / abs(da) + 1) + 1
    return bit_dict


# ---------------------------------------------------------------------------
# LR scheduler with warmup
# ---------------------------------------------------------------------------
class WarmupThenScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_steps, after_scheduler, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.warmup_steps > 0 and self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        if not self.finished:
            self.after_scheduler.base_lrs = [
                base_lr * (self.warmup_steps + 1) / max(self.warmup_steps, 1)
                for base_lr in self.base_lrs
            ]
            self.finished = True
        return self.after_scheduler.get_last_lr()

    def step(self, epoch=None):
        if self.finished:
            self.after_scheduler.step(None if epoch is None else epoch - self.warmup_steps)
        else:
            return super().step(epoch)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main(cfg):
    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(
        out_dir,
        f"{cfg.model}_geta_s{cfg.sparsity}_bit{cfg.min_bit_wt}-{cfg.max_bit_wt}.log",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger.info("=" * 60)
    logger.info(f"GETA  —  {cfg.model} on {cfg.dataset.upper()}")
    logger.info("=" * 60)
    for k, v in vars(cfg).items():
        logger.info(f"  {k:30s}: {v}")
    logger.info("=" * 60)

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed(cfg.seed)
    logger.info(f"Device: {device}")

    # Data
    train_loader, val_loader = get_loaders(cfg)
    steps_per_epoch = len(train_loader)
    logger.info(f"Steps per epoch: {steps_per_epoch}")

    # ---------------------------------------------------------------------------
    # Ablation schedule overrides
    # ---------------------------------------------------------------------------
    eff_warmup_steps  = 3 * steps_per_epoch
    eff_proj_periods  = cfg.projection_periods
    eff_proj_epochs   = cfg.projection_epochs
    eff_prune_start   = cfg.pruning_start_epoch
    eff_prune_periods = cfg.pruning_periods
    eff_prune_epochs  = cfg.pruning_epochs
    eff_total_epochs  = cfg.epochs

    if cfg.skip_warmup:
        eff_warmup_steps = 0
        logger.info("Ablation: LR warm-up disabled (warmup_steps=0)")

    if cfg.skip_projection:
        eff_proj_periods = 0
        eff_proj_epochs  = 0
        eff_prune_start  = 0
        logger.info("Ablation: projection stage disabled, pruning starts at epoch 0")

    if cfg.skip_joint:
        eff_prune_periods = 0
        logger.info("Ablation: joint pruning disabled (pruning_periods=0, no sparsity induced)")

    if cfg.skip_cooldown:
        eff_total_epochs = eff_prune_start + eff_prune_epochs
        logger.info(f"Ablation: cool-down disabled, total epochs reduced to {eff_total_epochs}")

    # ---------------------------------------------------------------------------
    # Model + Quantization
    # ---------------------------------------------------------------------------
    model = build_model(cfg.model, in_channels=3, out_channels=1)
    quant_mode = (
        QuantizationMode.WEIGHT_AND_ACTIVATION if cfg.act_quant
        else QuantizationMode.WEIGHT_ONLY
    )
    model = model_to_quantize_model(model, quant_mode=quant_mode)
    logger.info(f"Quantization mode: {quant_mode.value}")

    dummy_input = torch.rand(1, 3, cfg.img_size, cfg.img_size).to(device)
    oto = OTO(model.to(device), dummy_input=dummy_input)

    # Record full model stats before compression
    full_macs = oto.compute_macs(in_million=True, layerwise=True)
    full_bops = oto.compute_bops(in_million=True, layerwise=True)
    full_bops["total"] = full_bops["total"] * 32 / 16  # hotfix
    full_num_params = oto.compute_num_params(in_million=True)
    full_avg_bit = oto.compute_average_bit_width()
    logger.info(
        f"Full model — MACs: {full_macs['total']:.2f} M, "
        f"BOPs: {full_bops['total']:.2f} M, "
        f"Params: {full_num_params:.4f} M, "
        f"Avg bit width: {full_avg_bit:.2f}"
    )

    # ---------------------------------------------------------------------------
    # GETA optimizer
    # ---------------------------------------------------------------------------
    optimizer = oto.geta(
        variant=cfg.variant,
        lr=cfg.lr,
        lr_quant=cfg.lr_quant,
        first_momentum=0.9,
        weight_decay=cfg.weight_decay,
        target_group_sparsity=cfg.sparsity,
        start_projection_step=cfg.projection_start_epoch * steps_per_epoch,
        projection_periods=eff_proj_periods,
        projection_steps=eff_proj_epochs * steps_per_epoch,
        start_pruning_step=eff_prune_start * steps_per_epoch,
        pruning_periods=eff_prune_periods,
        pruning_steps=eff_prune_epochs * steps_per_epoch,
        bit_reduction=cfg.bit_reduction,
        min_bit_wt=cfg.min_bit_wt,
        max_bit_wt=cfg.max_bit_wt,
    )

    # ---------------------------------------------------------------------------
    # LR scheduler
    # ---------------------------------------------------------------------------
    criterion = DiceBCELoss()
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=eff_total_epochs * steps_per_epoch, eta_min=0
    )
    lr_scheduler = WarmupThenScheduler(
        optimizer, warmup_steps=eff_warmup_steps, after_scheduler=cosine
    )

    # ---------------------------------------------------------------------------
    # Checkpoint resume
    # ---------------------------------------------------------------------------
    start_epoch = 0
    best_dice = 0.0
    best_epoch = 0

    ckpt_path = os.path.join(out_dir, "checkpoint.pt")
    if cfg.resume and os.path.isfile(ckpt_path):
        logger.info(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        cosine.load_state_dict(ckpt["cosine_scheduler"])
        lr_scheduler.finished = ckpt["warmup_finished"]
        lr_scheduler.last_epoch = ckpt["warmup_last_epoch"]
        start_epoch = ckpt["epoch"] + 1
        best_dice = ckpt["best_dice"]
        best_epoch = ckpt["best_epoch"]
        logger.info(f"Resumed at epoch {start_epoch}, best_dice so far: {best_dice:.4f}")
    elif cfg.resume:
        logger.warning(f"--resume set but no checkpoint found at {ckpt_path}, starting fresh")

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    for epoch in range(start_epoch, eff_total_epochs):
        model.train()
        running_loss, running_dice, n_batches = 0.0, 0.0, 0

        for images, masks in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{eff_total_epochs}", leave=False
        ):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.grad_clipping()
            optimizer.step()
            lr_scheduler.step()

            running_loss += loss.item()
            running_dice += dice_coefficient(outputs, masks).item()
            n_batches += 1

        train_loss = running_loss / n_batches
        train_dice = running_dice / n_batches
        val_metrics = evaluate(model, val_loader, criterion, device)
        metrics = optimizer.compute_metrics()
        avg_bit = oto.compute_average_bit_width()

        logger.info(
            f"Epoch {epoch:>3d}/{eff_total_epochs} | "
            f"train_loss {train_loss:.4f} | train_dice {train_dice:.4f} | "
            f"val_dice {val_metrics['dice']:.4f} | val_iou {val_metrics['iou']:.4f} | "
            f"grp_sparsity {metrics.group_sparsity:.2f} | "
            f"avg_wt_bit {avg_bit:.2f} | "
            f"lr {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save(
                model.state_dict(),
                os.path.join(out_dir, f"{cfg.model}_geta_best.pt"),
            )

        # Save checkpoint every epoch (overwrites previous — cheap insurance)
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cosine_scheduler": cosine.state_dict(),
                "warmup_finished": lr_scheduler.finished,
                "warmup_last_epoch": lr_scheduler.last_epoch,
                "best_dice": best_dice,
                "best_epoch": best_epoch,
            },
            ckpt_path,
        )

    logger.info(f"Training done. Best val Dice: {best_dice:.4f} at epoch {best_epoch}")

    # ---------------------------------------------------------------------------
    # Construct compressed subnet
    # ---------------------------------------------------------------------------
    logger.info("Constructing compressed subnet...")
    oto.construct_subnet(out_dir=os.path.join(out_dir, "compressed"))
    compressed_model = torch.load(oto.compressed_model_path)
    oto_c = OTO(compressed_model, dummy_input)

    c_macs = oto_c.compute_macs(in_million=True, layerwise=True)
    c_bops = oto_c.compute_bops(in_million=True, layerwise=True)
    c_params = oto_c.compute_num_params(in_million=True)
    c_avg_bit = oto_c.compute_average_bit_width()

    logger.info("=" * 60)
    logger.info("COMPRESSION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Best val Dice         : {best_dice:.4f}")
    logger.info(f"  Full  MACs            : {full_macs['total']:.2f} M")
    logger.info(f"  Comp. MACs            : {c_macs['total']:.2f} M")
    logger.info(f"  Full  BOPs            : {full_bops['total']:.2f} M")
    logger.info(f"  Comp. BOPs            : {c_bops['total']:.2f} M")
    logger.info(f"  Rel. BOPs             : {c_bops['total'] / full_bops['total'] * 100:.2f}%")
    logger.info(f"  Full  params          : {full_num_params:.4f} M")
    logger.info(f"  Comp. params          : {c_params:.4f} M")
    logger.info(f"  Full  avg bit width   : {full_avg_bit:.2f}")
    logger.info(f"  Comp. avg bit width   : {c_avg_bit:.2f}")

    param_dict = get_quant_param_dict(model)
    bit_dict = get_bitwidth_dict(param_dict)
    logger.info("Per-layer bit widths:")
    logger.info(json.dumps(bit_dict, indent=2))
    logger.info("Done!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def get_config():
    p = argparse.ArgumentParser(description="GETA Medical Image Segmentation")

    # Data
    p.add_argument("--model", default="unet_small",
                   choices=["unet_tiny", "unet_small", "unet_standard", "unet_large",
                            "encdec_small", "encdec_standard"])
    p.add_argument("--dataset", default="lgg", choices=["lgg", "isic", "brats"],
                   help="Dataset: lgg (Brain MRI), isic (skin lesion), brats (BraTS 2020)")
    p.add_argument("--data_root", required=True,
                   help="Root data directory; loader appends lgg-mri-segmentation/ or isic2018/")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="./outputs/unet_geta")

    # Training
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_quant", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--variant", default="adam")

    # GETA compression schedule
    p.add_argument("--sparsity", type=float, default=0.3)
    p.add_argument("--projection_start_epoch", type=int, default=0)
    p.add_argument("--projection_periods", type=int, default=5)
    p.add_argument("--projection_epochs", type=int, default=15)
    p.add_argument("--pruning_start_epoch", type=int, default=15)
    p.add_argument("--pruning_periods", type=int, default=5)
    p.add_argument("--pruning_epochs", type=int, default=50)

    # Quantization
    p.add_argument("--bit_reduction", type=int, default=2)
    p.add_argument("--min_bit_wt", type=int, default=4)
    p.add_argument("--max_bit_wt", type=int, default=16)
    p.add_argument("--act_quant", action="store_true",
                   help="Enable weight + activation quantization (default: weight-only)")

    # Checkpoint resume
    p.add_argument("--resume", action="store_true",
                   help="Resume from checkpoint.pt in --out_dir if it exists")

    # QASSO ablation flags (disable one stage at a time)
    p.add_argument("--skip_warmup", action="store_true",
                   help="Ablation: disable LR warm-up (set warmup_steps=0)")
    p.add_argument("--skip_projection", action="store_true",
                   help="Ablation: disable projection stage (pruning starts immediately)")
    p.add_argument("--skip_joint", action="store_true",
                   help="Ablation: disable joint pruning (quantization only, no sparsity)")
    p.add_argument("--skip_cooldown", action="store_true",
                   help="Ablation: disable cool-down (stop after pruning ends)")

    return p.parse_args()


if __name__ == "__main__":
    main(get_config())
