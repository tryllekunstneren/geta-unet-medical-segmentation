import os
import glob
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class BrainMRIDataset(Dataset):

    def __init__(self, data_dir, img_size=256, augment=False):
        self.data_dir = data_dir
        self.img_size = img_size
        self.augment = augment

        # Find all mask files across all patient folders
        self.masks = sorted(glob.glob(os.path.join(data_dir, "*", "*_mask.tif")))
        # Derive corresponding image paths
        self.images = [m.replace("_mask.tif", ".tif") for m in self.masks]

        # Verify all images exist
        for img_path in self.images:
            if not os.path.exists(img_path):
                raise RuntimeError(f"Image not found: {img_path}")

        if len(self.images) == 0:
            raise RuntimeError(
                f"No images found in {data_dir}. "
                f"Check that kaggle_3m folder is in the correct location."
            )

        print(f"BrainMRI dataset: found {len(self.images)} image-mask pairs")

    def __len__(self):
        return len(self.images)

    def _augment(self, image, mask):
        """Apply random augmentations consistently to image and mask."""
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

        if random.random() > 0.5:
            image = T.ColorJitter(brightness=0.2, contrast=0.2)(image)

        return image, mask

    def __getitem__(self, idx):
        # Load image (3-channel) and mask (grayscale)
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")

        # Resize
        image = TF.resize(image, [self.img_size, self.img_size])
        mask = TF.resize(mask, [self.img_size, self.img_size],
                         interpolation=T.InterpolationMode.NEAREST)

        # Augmentation
        if self.augment:
            image, mask = self._augment(image, mask)

        # Convert to tensors
        image = TF.to_tensor(image)  # [3, H, W], float [0, 1]

        # Normalize per-image (z-score), as recommended for this dataset
        mean = image.mean(dim=(1, 2), keepdim=True)
        std = image.std(dim=(1, 2), keepdim=True) + 1e-8
        image = (image - mean) / std

        # Binarize mask
        mask = torch.as_tensor(np.array(mask), dtype=torch.float32)
        mask = (mask > 127.5).float()
        mask = mask.unsqueeze(0)  # [1, H, W]

        return image, mask


def get_brain_mri_loaders(data_root, img_size=256, batch_size=16, num_workers=4,
                           val_split=0.2, seed=0):

    kaggle_dir = os.path.join(data_root, "kaggle_3m")

    # Get patient directories (exclude files like data.csv, README)
    patients = sorted([
        d for d in os.listdir(kaggle_dir)
        if os.path.isdir(os.path.join(kaggle_dir, d))
    ])

    # Split at patient level (prevents data leakage)
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_val = int(len(patients) * val_split)
    val_patients = set(patients[:n_val])
    train_patients = set(patients[n_val:])

    print(f"Patients — train: {len(train_patients)}, val: {len(val_patients)}")

    # Create datasets
    full_dataset_train = BrainMRIDataset(kaggle_dir, img_size=img_size, augment=True)
    full_dataset_val = BrainMRIDataset(kaggle_dir, img_size=img_size, augment=False)

    # Find indices belonging to each split based on patient folder
    train_indices = []
    val_indices = []
    for idx, img_path in enumerate(full_dataset_train.images):
        # Extract patient folder name from path
        patient = os.path.basename(os.path.dirname(img_path))
        if patient in train_patients:
            train_indices.append(idx)
        else:
            val_indices.append(idx)

    train_subset = Subset(full_dataset_train, train_indices)
    val_subset = Subset(full_dataset_val, val_indices)

    print(f"Slices — train: {len(train_indices)}, val: {len(val_indices)}")

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader
