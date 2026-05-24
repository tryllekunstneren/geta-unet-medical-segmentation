import glob
import os

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ISICDataset(Dataset):

    def __init__(self, image_dir, mask_dir, img_size=256, augment=False):
        self.img_size = img_size
        self.augment = augment

        # Collect all mask files, derive image paths from them
        mask_files = sorted(glob.glob(os.path.join(mask_dir, "*_segmentation.png")))
        if len(mask_files) == 0:
            raise RuntimeError(
                f"No mask files found in {mask_dir}. "
                f"Expected files matching '*_segmentation.png'."
            )

        self.image_paths = []
        self.mask_paths = []
        for mask_path in mask_files:
            # ISIC_XXXXXXX_segmentation.png → ISIC_XXXXXXX.jpg
            basename = os.path.basename(mask_path)
            image_id = basename.replace("_segmentation.png", "")
            image_path = os.path.join(image_dir, image_id + ".jpg")
            if not os.path.exists(image_path):
                raise RuntimeError(f"Image not found for mask {mask_path}: expected {image_path}")
            self.image_paths.append(image_path)
            self.mask_paths.append(mask_path)

        print(f"ISICDataset: found {len(self.image_paths)} image-mask pairs in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def _augment(self, image, mask):
        """Apply identical random augmentations to image and mask."""
        import random
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
            image = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)(image)
        return image, mask

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        # Resize
        image = TF.resize(image, [self.img_size, self.img_size])
        mask = TF.resize(
            mask,
            [self.img_size, self.img_size],
            interpolation=T.InterpolationMode.NEAREST,
        )

        if self.augment:
            image, mask = self._augment(image, mask)

        # Image → float tensor, z-score normalise per image
        image = TF.to_tensor(image)  # [3, H, W] in [0, 1]
        mean = image.mean(dim=(1, 2), keepdim=True)
        std = image.std(dim=(1, 2), keepdim=True) + 1e-8
        image = (image - mean) / std

        # Mask → binary float [0, 1]
        mask = torch.as_tensor(np.array(mask), dtype=torch.float32)
        mask = (mask > 127.5).float()
        mask = mask.unsqueeze(0)  # [1, H, W]

        return image, mask


def get_isic_loaders(
    data_root,
    img_size=256,
    batch_size=16,
    num_workers=4,
):
    
    train_img_dir = os.path.join(data_root, "ISIC2018_Task1-2_Training_Input")
    train_mask_dir = os.path.join(data_root, "ISIC2018_Task1_Training_GroundTruth")
    val_img_dir = os.path.join(data_root, "ISIC2018_Task1-2_Validation_Input")
    val_mask_dir = os.path.join(data_root, "ISIC2018_Task1_Validation_GroundTruth")

    for d in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        if not os.path.isdir(d):
            raise RuntimeError(
                f"Expected directory not found: {d}\n"
                f"Check that data_root points to the isic2018/ folder."
            )

    train_dataset = ISICDataset(train_img_dir, train_mask_dir, img_size=img_size, augment=True)
    val_dataset = ISICDataset(val_img_dir, val_mask_dir, img_size=img_size, augment=False)

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
