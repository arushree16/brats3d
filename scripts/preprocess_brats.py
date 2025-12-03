#!/usr/bin/env python3
"""
Preprocess BraTS 2023 GLI cases:
- load NIfTI -> numpy
- optional N4 bias correction (SimpleITK)
- z-score per-modality (using brain voxels)
- center crop / pad to target_size (int or tuple)
- save to .npz (image: float32 [4,D,H,W], mask: uint8 [D,H,W])
"""

import os
import glob
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, Union
import argparse

# Optional: SimpleITK for N4
try:
    import SimpleITK as sitk
    HAVE_SITK = True
except Exception:
    HAVE_SITK = False
    # print("SimpleITK not available: N4 bias correction disabled")

def find_case_folders(base_dir: str):
    """Return list of case folders that contain BraTS-GLI-*"""
    p = Path(base_dir)
    folders = sorted([str(x) for x in p.glob("**/BraTS-GLI-*") if x.is_dir()])
    return folders

def load_nifti(path: str) -> np.ndarray:
    return nib.load(path).get_fdata(dtype=np.float32)

def n4_bias_field_correction(img_nib_path: str) -> np.ndarray:
    """Apply N4 bias correction using SimpleITK. Input file path."""
    if not HAVE_SITK:
        raise RuntimeError("SimpleITK not installed. pip install SimpleITK to use N4.")
    sitk_img = sitk.ReadImage(str(img_nib_path))
    maskImage = sitk.OtsuThreshold(sitk_img, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(sitk_img, maskImage)
    arr = sitk.GetArrayFromImage(corrected)  # z,y,x
    # BraTS nib loads as (x,y,z) so transpose
    return np.transpose(arr, (2,1,0)).astype(np.float32)

def zscore_normalize(volume: np.ndarray, mask: np.ndarray = None, clip: bool = True) -> np.ndarray:
    """
    Volume: np.ndarray (H,W,D) or (X,Y,Z)
    If mask provided, compute mean/std over mask>0.
    """
    v = volume.astype(np.float32)
    if mask is not None:
        brain_voxels = v[mask > 0]
        if brain_voxels.size == 0:
            mean = v.mean()
            std = v.std()
        else:
            mean = brain_voxels.mean()
            std = brain_voxels.std()
    else:
        mean = v.mean()
        std = v.std()
    if std == 0:
        std = 1.0
    out = (v - mean) / std
    if clip:
        out = np.clip(out, -6, 6)
    return out.astype(np.float32)

def center_crop_or_pad(img: np.ndarray, target: Union[int, Tuple[int,int,int]]) -> np.ndarray:
    """
    img: (X,Y,Z)
    target: int or (tx,ty,tz)
    returns center-cropped/padded array
    """
    if isinstance(target, int):
        tx = ty = tz = target
    else:
        tx, ty, tz = target

    x,y,z = img.shape
    # if crop
    startx = max(0, (x - tx)//2)
    starty = max(0, (y - ty)//2)
    startz = max(0, (z - tz)//2)
    cropped = img[startx: startx + tx, starty: starty + ty, startz: startz + tz]

    # pad if needed
    pad_x = max(0, tx - cropped.shape[0])
    pad_y = max(0, ty - cropped.shape[1])
    pad_z = max(0, tz - cropped.shape[2])

    pad_before = (pad_x//2, pad_y//2, pad_z//2)
    pad_after  = (pad_x - pad_before[0], pad_y - pad_before[1], pad_z - pad_before[2])

    padded = np.pad(
        cropped,
        (
            (pad_before[0], pad_after[0]),
            (pad_before[1], pad_after[1]),
            (pad_before[2], pad_after[2])
        ),
        mode='constant', constant_values=0
    )
    return padded

def preprocess_case(case_folder: str, out_folder: str, target_size=128, do_n4=False, save_torch=False):
    """
    case_folder: path to folder that contains files like:
        BraTS-GLI-XXXXX-t1n.nii.gz, -t1c.nii.gz, -t2f.nii.gz, -t2w.nii.gz, -seg.nii.gz
    Saves: out_folder/<case_id>.npz  with 'image' and 'mask'
    """
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    case_id = Path(case_folder).name
    # expected suffixes for BraTS2023
    files_map = {
        "t1n": glob.glob(os.path.join(case_folder, f"{case_id}*t1n*.nii*")),
        "t1c": glob.glob(os.path.join(case_folder, f"{case_id}*t1c*.nii*")),
        "t2f": glob.glob(os.path.join(case_folder, f"{case_id}*t2f*.nii*")),  # FLAIR
        "t2w": glob.glob(os.path.join(case_folder, f"{case_id}*t2w*.nii*")),
        "seg": glob.glob(os.path.join(case_folder, f"{case_id}*seg*.nii*")),
    }

    # basic check
    for k, v in files_map.items():
        if len(v) == 0 and k != "seg":
            raise FileNotFoundError(f"Could not find modality {k} in {case_folder}. Found: {v}")
    # load arrays
    if do_n4:
        # N4 per modality (expensive)
        modalities = {}
        for k in ("t1n","t1c","t2f","t2w"):
            modalities[k] = n4_bias_field_correction(files_map[k][0])
    else:
        modalities = {k: load_nifti(files_map[k][0]) for k in ("t1n","t1c","t2f","t2w")}
    mask = load_nifti(files_map["seg"][0]).astype(np.uint8)

    # reorder if necessary: nibabel likely returns (X,Y,Z)
    # Normalize each modality using brain voxels (mask>0)
    normed = []
    for k in ("t1n","t1c","t2f","t2w"):
        normed_mod = zscore_normalize(modalities[k], mask=mask)
        normed.append(normed_mod)

    # stack channels: we want shape [C, X, Y, Z] or [4, D, H, W]? we'll save as [4, X, Y, Z]
    image = np.stack(normed, axis=0).astype(np.float32)
    # center crop/pad each channel and mask
    C, X, Y, Z = image.shape
    # apply crop/pad per channel
    cropped_channels = np.zeros((C,)+center_crop_or_pad(image[0], target_size).shape, dtype=np.float32)
    for c in range(C):
        cropped_channels[c] = center_crop_or_pad(image[c], target_size)
    cropped_mask = center_crop_or_pad(mask, target_size).astype(np.uint8)

    out_path = os.path.join(out_folder, f"{case_id}.npz")
    np.savez_compressed(out_path, image=cropped_channels, mask=cropped_mask)
    if save_torch:
        try:
            import torch
            torch.save({"image": cropped_channels, "mask": cropped_mask}, out_path.replace(".npz", ".pt"))
        except Exception:
            pass
    return out_path

def bulk_preprocess(base_dir: str, out_dir: str, target_size=128, do_n4=False, max_cases=None):
    folders = find_case_folders(base_dir)
    print(f"Found {len(folders)} cases")
    if max_cases:
        folders = folders[:max_cases]
    for i, case in enumerate(folders, 1):
        print(f"[{i}/{len(folders)}] Preprocessing {case}")
        try:
            out = preprocess_case(case, out_dir, target_size=target_size, do_n4=do_n4)
            print("Saved:", out)
        except Exception as e:
            print("Failed:", case, e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="BraTS train base folder that contains ASNR-MICCAI-... directory")
    parser.add_argument("--out", required=True, help="output folder for preprocessed .npz")
    parser.add_argument("--size", type=int, default=128, help="target cubic size (default 128)")
    parser.add_argument("--n4", action="store_true", help="apply N4 bias correction (slow)")
    parser.add_argument("--max", type=int, default=None, help="limit number of cases")
    args = parser.parse_args()
    bulk_preprocess(args.base, args.out, target_size=args.size, do_n4=args.n4, max_cases=args.max)
'''python scripts/preprocess_brats.py \
  --base "/Users/arushreemishra/Downloads/3d brain tumor/data/raw/BraTS-GLI/train/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData" \
  --out "/Users/arushreemishra/Downloads/3d brain tumor/data/processed/brats128" \
  --size 128 \
  --max 20
'''
#processing only 20/1251 3d volumes due to storage issues