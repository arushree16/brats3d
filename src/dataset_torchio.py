"""
BRATSDataset using preprocessed .npz files and TorchIO augmentations.
Returns:
    x: torch.FloatTensor [4, D, H, W]
    y: torch.LongTensor [D, H, W]
"""

import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

# TorchIO usage
try:
    import torchio as tio
    HAVE_TORCHIO = True
except Exception:
    HAVE_TORCHIO = False
    # print("torchio not installed. Install via `pip install torchio` for augmentations.")

class BRATSDataset(Dataset):
    def __init__(self, preproc_folder: str, split='train', transforms=None):
        """
        preproc_folder: folder with .npz files (image, mask)
        transforms: torchio transforms (applied to Subject)
        """
        self.folder = Path(preproc_folder)
        self.files = sorted(list(self.folder.glob("*.npz")))
        self.transforms = transforms
        if len(self.files) == 0:
            raise RuntimeError(f"No .npz files found in {preproc_folder}")

    def __len__(self):
        return len(self.files)

    def _load_npz(self, path: Path):
        data = np.load(str(path))
        image = data['image']   # (4, X, Y, Z)
        mask = data['mask']     # (X, Y, Z)
        return image.astype(np.float32), mask.astype(np.int64)

    def __getitem__(self, idx):
        path = self.files[idx]
        image, mask = self._load_npz(path)
        # to torch
        image_t = torch.from_numpy(image)  # float32 [4,X,Y,Z]
        mask_t = torch.from_numpy(mask).long()  # [X,Y,Z]
        if HAVE_TORCHIO and self.transforms is not None:
            # Convert to TorchIO Subject
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image_t),   # channels-first
                mask=tio.LabelMap(tensor=mask_t.unsqueeze(0))
            )
            subject = self.transforms(subject)
            image_t = subject['image'].data  # [C, X, Y, Z]
            mask_t = subject['mask'].data.squeeze(0).long()  # [X,Y,Z]
        return image_t, mask_t

def get_default_augmentations(patch_size=(128,128,128)):
    if not HAVE_TORCHIO:
        return None
    transforms = [
        tio.RandomFlip(axes=('LR',), p=0.5),
        tio.RandomAffine(scales=(0.9, 1.1), degrees=10, translation=5, p=0.4),
        tio.RandomElasticDeformation(num_control_points=7, max_displacement=7.5, p=0.25),
        tio.RandomNoise(p=0.25),
        tio.RandomBiasField(p=0.2),   # synthetic bias
        tio.RandomGamma(p=0.2),
        tio.OneOf({tio.RandomBlur(): 0.1, tio.RandomMotion(): 0.1}, p=0.1),
        tio.Resample(1.0)  # if you want to resample to isotropic 1mm (optional)
    ]
    # Spatial crop/patch augmentation (if you'd like patch-based training)
    # Optionally include RandomSpatialCrop if patching smaller than whole volume
    composed = tio.Compose(transforms)
    return composed

def make_loaders(preproc_folder, batch_size=1, num_workers=2, shuffle_train=True, augment=True):
    if augment and HAVE_TORCHIO:
        transforms = get_default_augmentations()
    else:
        transforms = None
    ds = BRATSDataset(preproc_folder, transforms=transforms if shuffle_train else None)
    # split into train/val simple split (80/20)
    n = len(ds)
    idxs = list(range(n))
    random.seed(42)
    random.shuffle(idxs)
    split = int(0.8 * n)
    train_idx, val_idx = idxs[:split], idxs[split:]
    from torch.utils.data.sampler import SubsetRandomSampler
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    
    # Colab optimization: reduce num_workers to prevent overhead
    optimal_workers = 0 if batch_size == 1 else min(2, num_workers)
    
    train_loader = DataLoader(ds, batch_size=batch_size, sampler=train_sampler,
                              num_workers=optimal_workers, pin_memory=True, 
                              persistent_workers=True if optimal_workers > 0 else False)
    val_loader = DataLoader(ds, batch_size=batch_size, sampler=val_sampler,
                            num_workers=optimal_workers, pin_memory=True,
                            persistent_workers=True if optimal_workers > 0 else False)
    return train_loader, val_loader
