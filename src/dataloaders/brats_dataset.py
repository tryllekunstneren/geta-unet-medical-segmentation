import glob
import os
import random

import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


class BratsDataset(Dataset):

    def __init__(self, data_dir, volume_ids, img_size=256, augment=False):
        self.data_dir = data_dir
        self.img_size = img_size
        self.augment = augment

        # Collect all .h5 files belonging to the requested volumes
        # and filter out empty slices (no tumor pixels)
        self.file_paths = []
        volume_set = set(volume_ids)

        all_h5 = sorted(glob.glob(os.path.join(data_dir, "volume_*_slice_*.h5")))
        if len(all_h5) == 0:
            raise RuntimeError(
                f"No .h5 files found in {data_dir}. "
                f"Check that data_dir points to BraTS2020_training_data/."
            )

        for path in all_h5:
            # Parse volume index from filename: volume_<N>_slice_<M>.h5
            basename = os.path.basename(path)
            try:
                vol_idx = int(basename.split("_")[1])
            except (IndexError, ValueError):
                continue
            if vol_idx not in volume_set:
                continue

            # Filter: only keep slices with at least one tumor pixel
            try:
                with h5py.File(path, "r") as f:
                    mask = f["mask"][()]          # (240, 240, 3) uint8
                binary = mask.any(axis=-1)        # (240, 240) bool
                if binary.sum() == 0:
                    continue
            except Exception:
                continue

            self.file_paths.append(path)

        print(
            f"BraTSDataset: {len(self.file_paths)} non-empty slices "
            f"from {len(volume_ids)} volumes in {data_dir}"
        )

    def __len__(self):
        return len(self.file_paths)

    def _augment(self, image, mask):
        """Apply identical random augmentations to image and mask PIL tensors."""
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        return image, mask

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        with h5py.File(path, "r") as f:
            image_raw = f["image"][()]    # (240, 240, 4) float64
            mask_raw = f["mask"][()]      # (240, 240, 3) uint8

        # Use FLAIR channel only (index 3), cast to float32
        flair = image_raw[..., 3].astype(np.float32)    # (240, 240)

        # Stack FLAIR × 3 → (3, 240, 240) to match LGG's 3-channel RGB input
        image_np = np.stack([flair, flair, flair], axis=0)  # (3, 240, 240)

        # Binary mask: any of the 3 tumor classes → 1
        binary_mask = mask_raw.any(axis=-1).astype(np.float32)  # (240, 240)

        # Convert to torch tensors
        image = torch.from_numpy(image_np)           # (3, 240, 240)
        mask = torch.from_numpy(binary_mask)         # (240, 240)

        # Resize from 240×240 → img_size×img_size using bilinear/nearest
        image = TF.resize(image, [self.img_size, self.img_size])
        mask = mask.unsqueeze(0)                     # (1, 240, 240)
        mask = TF.resize(
            mask,
            [self.img_size, self.img_size],
            interpolation=T.InterpolationMode.NEAREST,
        )

        # Augmentation (operate on tensors directly — no PIL needed)
        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

        # Z-score normalise per slice (same as LGG)
        mean = image.mean(dim=(1, 2), keepdim=True)
        std = image.std(dim=(1, 2), keepdim=True) + 1e-8
        image = (image - mean) / std

        # Ensure mask is binary float [0, 1]
        mask = (mask > 0.5).float()

        return image, mask  # (3, H, W), (1, H, W)


def get_brats_loaders(
    data_root,
    img_size=256,
    batch_size=16,
    num_workers=4,
    val_split=0.2,
    seed=0,
):
    
    # Discover all unique volume indices from filenames
    all_h5 = sorted(glob.glob(os.path.join(data_root, "volume_*_slice_*.h5")))
    if len(all_h5) == 0:
        raise RuntimeError(
            f"No .h5 files found in {data_root}. "
            f"Expected files like volume_<N>_slice_<M>.h5."
        )

    volume_ids = sorted(set(
        int(os.path.basename(p).split("_")[1]) for p in all_h5
    ))
    print(f"BraTS: found {len(volume_ids)} unique volumes ({len(all_h5)} total slices)")

    # Reproducible patient-level split
    rng = random.Random(seed)
    shuffled = volume_ids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_split))
    val_volumes = set(shuffled[:n_val])
    train_volumes = set(shuffled[n_val:])
    print(f"BraTS split — train: {len(train_volumes)} volumes, val: {len(val_volumes)} volumes")

    train_dataset = BratsDataset(data_root, list(train_volumes), img_size=img_size, augment=True)
    val_dataset = BratsDataset(data_root, list(val_volumes), img_size=img_size, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
