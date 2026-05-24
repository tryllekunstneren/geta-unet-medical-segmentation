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

warnings.filterwarnings("ignore")
logger = logging.getLogger("sequential")


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
# Metrics & Loss (identical to baseline/geta for fair comparison)
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
# Quantization parameter utilities (reused from geta script)
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
    return bit_dict


# ---------------------------------------------------------------------------
# Phase 1: Structured pruning with HESSO (no quantization)
# ---------------------------------------------------------------------------
def run_phase1(cfg, train_loader, val_loader, device):
    logger.info("=" * 60)
    logger.info("PHASE 1 — Structured Pruning (HESSO, no quantization)")
    logger.info("=" * 60)

    phase1_dir = os.path.join(cfg.out_dir, "phase1")
    os.makedirs(phase1_dir, exist_ok=True)

    steps_per_epoch = len(train_loader)
    model = build_model(cfg.model, in_channels=3, out_channels=1)
    dummy_input = torch.rand(1, 3, cfg.img_size, cfg.img_size).to(device)

    oto = OTO(model.to(device), dummy_input=dummy_input)

    full_macs = oto.compute_macs(in_million=True)
    full_params = oto.compute_num_params(in_million=True)
    logger.info(f"Full model — MACs: {full_macs['total']:.2f} M, Params: {full_params:.4f} M")

    optimizer = oto.hesso(
        variant=cfg.variant,
        lr=cfg.lr,
        first_momentum=0.9,
        weight_decay=cfg.weight_decay,
        target_group_sparsity=cfg.sparsity,
        start_pruning_step=cfg.pruning_start_epoch * steps_per_epoch,
        pruning_periods=cfg.pruning_periods,
        pruning_steps=cfg.pruning_epochs * steps_per_epoch,
    )

    criterion = DiceBCELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.phase1_epochs * steps_per_epoch, eta_min=0
    )

    best_dice = 0.0
    best_epoch = 0

    for epoch in range(cfg.phase1_epochs):
        model.train()
        running_loss, running_dice, n_batches = 0.0, 0.0, 0

        for images, masks in tqdm(train_loader, desc=f"[P1] Epoch {epoch+1}/{cfg.phase1_epochs}", leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            # HESSO does not have grad_clipping; use standard clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            running_dice += dice_coefficient(outputs, masks).item()
            n_batches += 1

        val_metrics = evaluate(model, val_loader, criterion, device)
        train_loss = running_loss / n_batches
        train_dice = running_dice / n_batches

        logger.info(
            f"[P1] Epoch {epoch:>3d}/{cfg.phase1_epochs} | "
            f"train_loss {train_loss:.4f} | train_dice {train_dice:.4f} | "
            f"val_dice {val_metrics['dice']:.4f} | val_iou {val_metrics['iou']:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(phase1_dir, "pruned_best.pt"))

    logger.info(f"Phase 1 done. Best val Dice: {best_dice:.4f} at epoch {best_epoch}")

    # Construct pruned subnet
    logger.info("Constructing pruned subnet...")
    oto.construct_subnet(out_dir=phase1_dir)
    pruned_model_path = oto.compressed_model_path

    oto_c = OTO(torch.load(pruned_model_path), dummy_input)
    c_macs = oto_c.compute_macs(in_million=True)
    c_params = oto_c.compute_num_params(in_million=True)
    logger.info(f"Pruned model — MACs: {c_macs['total']:.2f} M  "
                f"({c_macs['total']/full_macs['total']*100:.1f}%), "
                f"Params: {c_params:.4f} M  "
                f"({c_params/full_params*100:.1f}%)")

    return pruned_model_path, best_dice, full_macs["total"], c_macs["total"], full_params, c_params


# ---------------------------------------------------------------------------
# Phase 2: Quantization-aware fine-tuning of the pruned model
# ---------------------------------------------------------------------------
def run_phase2(cfg, pruned_model_path, train_loader, val_loader, device):
    logger.info("=" * 60)
    logger.info("PHASE 2 — Quantization-Aware Fine-Tuning")
    logger.info("=" * 60)

    phase2_dir = os.path.join(cfg.out_dir, "phase2")
    os.makedirs(phase2_dir, exist_ok=True)

    dummy_input = torch.rand(1, 3, cfg.img_size, cfg.img_size).to(device)

    # Load pruned model and apply quantization
    pruned_model = torch.load(pruned_model_path)
    pruned_model = model_to_quantize_model(pruned_model)
    pruned_model = pruned_model.to(device)

    # Separate quantization parameters from regular weights for different LRs
    quant_param_names = {"d_quant", "t_quant", "q_m"}
    quant_params, regular_params = [], []
    for name, param in pruned_model.named_parameters():
        if any(k in name for k in quant_param_names):
            quant_params.append(param)
        else:
            regular_params.append(param)

    logger.info(f"Phase 2 param groups — regular: {len(regular_params)}, quant: {len(quant_params)}")

    optimizer = torch.optim.Adam([
        {"params": regular_params, "lr": cfg.lr_finetune},
        {"params": quant_params,   "lr": cfg.lr_quant},
    ], weight_decay=cfg.weight_decay)

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.phase2_epochs * steps_per_epoch, eta_min=0
    )

    criterion = DiceBCELoss()
    best_dice = 0.0
    best_epoch = 0

    # Track initial average bit width
    oto_p2 = OTO(pruned_model, dummy_input)
    logger.info(f"Avg bit width at phase 2 start: {oto_p2.compute_average_bit_width():.2f}")

    for epoch in range(cfg.phase2_epochs):
        pruned_model.train()
        running_loss, running_dice, n_batches = 0.0, 0.0, 0

        for images, masks in tqdm(train_loader, desc=f"[P2] Epoch {epoch+1}/{cfg.phase2_epochs}", leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = pruned_model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pruned_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            running_dice += dice_coefficient(outputs, masks).item()
            n_batches += 1

        val_metrics = evaluate(pruned_model, val_loader, criterion, device)
        train_loss = running_loss / n_batches
        train_dice = running_dice / n_batches
        avg_bit = oto_p2.compute_average_bit_width()

        logger.info(
            f"[P2] Epoch {epoch:>3d}/{cfg.phase2_epochs} | "
            f"train_loss {train_loss:.4f} | train_dice {train_dice:.4f} | "
            f"val_dice {val_metrics['dice']:.4f} | val_iou {val_metrics['iou']:.4f} | "
            f"avg_wt_bit {avg_bit:.2f} | "
            f"lr {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save(pruned_model.state_dict(), os.path.join(phase2_dir, "quantized_best.pt"))

    logger.info(f"Phase 2 done. Best val Dice: {best_dice:.4f} at epoch {best_epoch}")

    final_avg_bit = oto_p2.compute_average_bit_width()
    param_dict = get_quant_param_dict(pruned_model)
    bit_dict = get_bitwidth_dict(param_dict)
    logger.info(f"Final avg bit width: {final_avg_bit:.2f}")
    logger.info("Per-layer bit widths:")
    logger.info(json.dumps(bit_dict, indent=2))

    return best_dice, final_avg_bit


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)

    log_file = os.path.join(cfg.out_dir, f"{cfg.model}_sequential_s{cfg.sparsity}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger.info("=" * 60)
    logger.info(f"Sequential Baseline  —  {cfg.model} on Brain MRI")
    logger.info(f"  Phase 1: {cfg.phase1_epochs} epochs HESSO pruning (sparsity={cfg.sparsity})")
    logger.info(f"  Phase 2: {cfg.phase2_epochs} epochs QAT fine-tuning")
    logger.info("=" * 60)
    for k, v in vars(cfg).items():
        logger.info(f"  {k:30s}: {v}")
    logger.info("=" * 60)

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed(cfg.seed)
    logger.info(f"Device: {device}")

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

    # Phase 1: prune
    pruned_model_path, p1_best_dice, full_macs, pruned_macs, full_params, pruned_params = run_phase1(
        cfg, train_loader, val_loader, device
    )

    # Phase 2: quantize
    p2_best_dice, final_avg_bit = run_phase2(
        cfg, pruned_model_path, train_loader, val_loader, device
    )

    # Final summary
    logger.info("=" * 60)
    logger.info("SEQUENTIAL COMPRESSION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Model                  : {cfg.model}")
    logger.info(f"  Sparsity target        : {cfg.sparsity}")
    logger.info(f"  Phase 1 best val Dice  : {p1_best_dice:.4f}  (after pruning)")
    logger.info(f"  Phase 2 best val Dice  : {p2_best_dice:.4f}  (after QAT)")
    logger.info(f"  Full  MACs             : {full_macs:.2f} M")
    logger.info(f"  Pruned MACs            : {pruned_macs:.2f} M  ({pruned_macs/full_macs*100:.1f}%)")
    logger.info(f"  Full  params           : {full_params:.4f} M")
    logger.info(f"  Pruned params          : {pruned_params:.4f} M  ({pruned_params/full_params*100:.1f}%)")
    logger.info(f"  Final avg bit width    : {final_avg_bit:.2f}")
    logger.info("Done!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def get_config():
    p = argparse.ArgumentParser(description="Sequential Prune-then-Quantize baseline")

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
    p.add_argument("--out_dir", default="./outputs/sequential")

    # Optimizer
    p.add_argument("--variant", default="adam")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate for phase 1 (pruning)")
    p.add_argument("--lr_finetune", type=float, default=1e-4,
                   help="Learning rate for phase 2 weight fine-tuning (lower since pre-trained)")
    p.add_argument("--lr_quant", type=float, default=1e-3,
                   help="Learning rate for quantization parameters in phase 2")
    p.add_argument("--weight_decay", type=float, default=1e-4)

    # Phase 1: pruning schedule
    p.add_argument("--sparsity", type=float, default=0.3)
    p.add_argument("--phase1_epochs", type=int, default=100)
    p.add_argument("--pruning_start_epoch", type=int, default=10)
    p.add_argument("--pruning_periods", type=int, default=5)
    p.add_argument("--pruning_epochs", type=int, default=70,
                   help="Number of epochs over which pruning is spread")

    # Phase 2: QAT fine-tuning
    p.add_argument("--phase2_epochs", type=int, default=50)

    return p.parse_args()


if __name__ == "__main__":
    main(get_config())
