import argparse
import json
import logging
import math
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from only_train_once import OTO
from only_train_once.quantization.quant_model import model_to_quantize_model
from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10

# Ignore warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("geta_resnet20")


# ---------------------------------------------------------------------------
# Learning rate scheduler with warmup
# ---------------------------------------------------------------------------
class WarmupThenScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Linearly warm up LR, then hand off to another scheduler."""

    def __init__(self, optimizer, warmup_steps, after_scheduler, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        if not self.finished:
            self.after_scheduler.base_lrs = [
                base_lr * (self.warmup_steps + 1) / self.warmup_steps
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
# Utility: check test accuracy
# ---------------------------------------------------------------------------
@torch.no_grad()
def check_accuracy(model, loader, device):
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, pred = outputs.topk(5, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        correct1 += correct[:1].reshape(-1).float().sum(0).item()
        correct5 += correct[:5].reshape(-1).float().sum(0).item()
        total += targets.size(0)
    acc1 = 100.0 * correct1 / total
    acc5 = 100.0 * correct5 / total
    return acc1, acc5


# ---------------------------------------------------------------------------
# Utility: quantization parameter inspection
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
            d_a = vals["d_quant_act"]
            q_a = abs(vals["q_m_act"])
            t_a = vals.get("t_quant_act", 1.0)
            bit_dict[key]["activation"] = (
                math.log2(math.exp(t_a * math.log(q_a)) / abs(d_a) + 1) + 1
            )
    return bit_dict


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def get_data_loaders(batch_size, num_workers):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    data_root = os.environ.get("DATA_DIR", "./data/cifar10")
    trainset = CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    testset = CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main(cfg):
    # ---- Output directories ----
    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # ---- Logging ----
    log_file = os.path.join(
        out_dir, f"resnet20_s{cfg.sparsity}_bit{cfg.min_bit_wt}-{cfg.max_bit_wt}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger.info("=" * 60)
    logger.info("GETA  —  ResNet20 on CIFAR10")
    logger.info("=" * 60)
    for k, v in vars(cfg).items():
        logger.info(f"  {k:30s}: {v}")
    logger.info("=" * 60)

    # ---- Seed ----
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed(cfg.seed)
    logger.info(f"Device: {device}")

    # ---- Data ----
    train_loader, test_loader = get_data_loaders(cfg.batch_size, cfg.num_workers)
    steps_per_epoch = len(train_loader)
    logger.info(f"Steps per epoch: {steps_per_epoch}")

    # ---- Model + Quantization ----
    model = resnet20_cifar10()
    model = model_to_quantize_model(model)  # weight-only quantization by default
    dummy_input = torch.rand(1, 3, 32, 32).to(device)

    # ---- OTO / GETA setup ----
    oto = OTO(model.to(device), dummy_input=dummy_input)

    # Sanity check: the assertion from the original script
    assert cfg.pruning_start_epoch == cfg.projection_start_epoch + cfg.projection_epochs, (
        f"pruning_start_epoch ({cfg.pruning_start_epoch}) must equal "
        f"projection_start_epoch + projection_epochs "
        f"({cfg.projection_start_epoch} + {cfg.projection_epochs})"
    )

    optimizer = oto.geta(
        variant=cfg.variant,
        lr=cfg.lr,
        lr_quant=cfg.lr_quant,
        first_momentum=0.9,
        weight_decay=cfg.weight_decay,
        target_group_sparsity=cfg.sparsity,
        # Projection stage
        start_projection_step=cfg.projection_start_epoch * steps_per_epoch,
        projection_periods=cfg.projection_periods,
        projection_steps=cfg.projection_epochs * steps_per_epoch,
        # Pruning (joint) stage
        start_pruning_step=cfg.pruning_start_epoch * steps_per_epoch,
        pruning_periods=cfg.pruning_periods,
        pruning_steps=cfg.pruning_epochs * steps_per_epoch,
        # Quantization bit width
        bit_reduction=cfg.bit_reduction,
        min_bit_wt=cfg.min_bit_wt,
        max_bit_wt=cfg.max_bit_wt,
    )

    # ---- Record full-model stats before training ----
    full_macs = oto.compute_macs(in_million=True, layerwise=True)
    full_bops = oto.compute_bops(in_million=True, layerwise=True)
    full_bops["total"] = full_bops["total"] * 32 / 16  # hotfix from original code
    full_num_params = oto.compute_num_params(in_million=True)
    full_avg_bit = oto.compute_average_bit_width()

    logger.info(f"Full model — MACs: {full_macs['total']:.2f} M, "
                f"BOPs: {full_bops['total']:.2f} M, "
                f"Params: {full_num_params:.4f} M, "
                f"Avg bit width: {full_avg_bit:.2f}")

    # ---- Loss + LR scheduler ----
    criterion = nn.CrossEntropyLoss()

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs * steps_per_epoch, eta_min=0
    )
    lr_scheduler = WarmupThenScheduler(
        optimizer, warmup_steps=5 * steps_per_epoch, after_scheduler=cosine_scheduler
    )

    # ---- Training ----
    best_acc1 = 0.0
    best_epoch = 0

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.grad_clipping()
            optimizer.step()
            lr_scheduler.step()

            running_loss += loss.item()

        # ---- Epoch-end evaluation ----
        avg_loss = running_loss / steps_per_epoch
        acc1, acc5 = check_accuracy(model, test_loader, device)
        metrics = optimizer.compute_metrics()
        avg_bit = oto.compute_average_bit_width()

        logger.info(
            f"Epoch {epoch:>3d}/{cfg.epochs} | "
            f"loss {avg_loss:.3f} | "
            f"acc1 {acc1:.2f}% | acc5 {acc5:.2f}% | "
            f"grp_sparsity {metrics.group_sparsity:.2f} | "
            f"avg_wt_bit {avg_bit:.2f} | "
            f"lr {optimizer.param_groups[0]['lr']:.6f}"
        )

        if acc1 > best_acc1:
            best_acc1 = acc1
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(out_dir, "resnet20_best.pt"))

    logger.info(f"Training done. Best acc1: {best_acc1:.2f}% at epoch {best_epoch}")

    # ---- Construct compressed subnet ----
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
    logger.info(f"  Best test accuracy    : {best_acc1:.2f}%")
    logger.info(f"  Full  MACs            : {full_macs['total']:.2f} M")
    logger.info(f"  Comp. MACs            : {c_macs['total']:.2f} M")
    logger.info(f"  Full  BOPs            : {full_bops['total']:.2f} M")
    logger.info(f"  Comp. BOPs            : {c_bops['total']:.2f} M")
    logger.info(f"  Rel. BOPs             : {c_bops['total'] / full_bops['total'] * 100:.2f}%")
    logger.info(f"  Full  params          : {full_num_params:.4f} M")
    logger.info(f"  Comp. params          : {c_params:.4f} M")
    logger.info(f"  Full  avg bit width   : {full_avg_bit:.2f}")
    logger.info(f"  Comp. avg bit width   : {c_avg_bit:.2f}")

    # ---- Per-layer bit widths ----
    param_dict = get_quant_param_dict(model)
    bit_dict = get_bitwidth_dict(param_dict)
    logger.info("Per-layer bit widths:")
    logger.info(json.dumps(bit_dict, indent=2))

    logger.info("Done!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def get_config():
    p = argparse.ArgumentParser(description="GETA ResNet20 CIFAR10")

    # Model / data
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="./outputs/resnet20_cifar10")

    # Training
    p.add_argument("--epochs", type=int, default=350,
                   help="Total training epochs (paper Table 7: 350)")
    p.add_argument("--lr", type=float, default=0.1,
                   help="Initial learning rate (paper Appendix C: 1e-1)")
    p.add_argument("--lr_quant", type=float, default=1e-3,
                   help="LR for quantization params (script default: 1e-3)")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--variant", default="sgd",
                   help="Optimizer variant: sgd or adam")

    # GETA compression schedule
    # From paper Table 7: B=7 projection periods, Kb=35 projection steps,
    #                      P=5 pruning periods, Kp=30 pruning steps
    # The schedule in epochs:
    #   warm-up:    epochs [0, projection_start_epoch)
    #   projection: epochs [projection_start_epoch, pruning_start_epoch)
    #   joint/prune: epochs [pruning_start_epoch, pruning_start_epoch + pruning_epochs)
    #   cool-down:  remaining epochs until 'epochs'
    p.add_argument("--sparsity", type=float, default=0.35,
                   help="Target group sparsity (paper Table 7: 0.35)")
    p.add_argument("--projection_start_epoch", type=int, default=0,
                   help="Epoch to begin projection stage")
    p.add_argument("--projection_periods", type=int, default=7,
                   help="Number of projection periods B (paper: 7)")
    p.add_argument("--projection_epochs", type=int, default=35,
                   help="Total epochs for projection stage = B * Kb_per_period")
    p.add_argument("--pruning_start_epoch", type=int, default=35,
                   help="Epoch to begin pruning stage (= proj_start + proj_epochs)")
    p.add_argument("--pruning_periods", type=int, default=5,
                   help="Number of pruning periods P (paper: 5)")
    p.add_argument("--pruning_epochs", type=int, default=150,
                   help="Total epochs for pruning stage = P * Kp_per_period")

    # Quantization
    p.add_argument("--bit_reduction", type=int, default=2,
                   help="Bit width reduction per projection period (paper: 2)")
    p.add_argument("--min_bit_wt", type=int, default=4,
                   help="Min weight bit width (paper: 4)")
    p.add_argument("--max_bit_wt", type=int, default=16,
                   help="Max weight bit width (paper: 16)")

    cfg = p.parse_args()
    return cfg


if __name__ == "__main__":
    main(get_config())
