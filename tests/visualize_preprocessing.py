import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG — EDIT THESE PATHS
# ----------------------------
raw_t1_path = "/Users/arushreemishra/Downloads/3d brain tumor/data/raw/BraTS-GLI/train/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1n.nii.gz"
raw_flair_path = "/Users/arushreemishra/Downloads/3d brain tumor/data/raw/BraTS-GLI/train/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2f.nii.gz"
npz_path = "/Users/arushreemishra/Downloads/3d brain tumor/data/processed/brats128/BraTS-GLI-00000-000.npz"
# ----------------------------

def load_nifti(path):
    """Load raw NIfTI as numpy."""
    return nib.load(path).get_fdata()

def show_slice(volume, title, sl=None, cmap="gray"):
    """Show a single slice from a 3D volume."""
    if sl is None:
        sl = volume.shape[2] // 2  # middle slice

    plt.imshow(volume[:, :, sl], cmap=cmap)
    plt.title(f"{title} (slice={sl})")
    plt.axis("off")

# ----------------------------
# LOAD RAW FILES
# ----------------------------
raw_t1 = load_nifti(raw_t1_path)
raw_flair = load_nifti(raw_flair_path)

# ----------------------------
# LOAD PREPROCESSED FILE
# ----------------------------
data = np.load(npz_path)
proc_img = data["image"]     # (4,128,128,128)
proc_mask = data["mask"]     # (128,128,128)

# Extract modalities from preprocessed
proc_t1   = proc_img[0]
proc_t1ce = proc_img[1]
proc_flair= proc_img[2]
proc_t2   = proc_img[3]

# ----------------------------
# DERIVE WT/TC/ET
# ----------------------------
WT = (proc_mask > 0)
TC = np.isin(proc_mask, [1,4])
ET = (proc_mask == 4)

# ----------------------------
# PLOT BEFORE/AFTER
# ----------------------------
plt.figure(figsize=(16,12))

# RAW T1
plt.subplot(3,3,1)
show_slice(raw_t1, "RAW T1 (nii.gz)", cmap="gray")

# RAW FLAIR
plt.subplot(3,3,2)
show_slice(raw_flair, "RAW FLAIR (nii.gz)", cmap="gray")

# RAW shape
plt.subplot(3,3,3)
plt.text(0.1, 0.5, f"RAW SHAPE = {raw_t1.shape}", fontsize=18)
plt.axis("off")

# PROCESSED T1
plt.subplot(3,3,4)
show_slice(proc_t1, "PROCESSED T1 (npz)", cmap="gray")

# PROCESSED FLAIR
plt.subplot(3,3,5)
show_slice(proc_flair, "PROCESSED FLAIR (npz)", cmap="gray")

# PROC shape
plt.subplot(3,3,6)
plt.text(0.1, 0.5, f"PROCESSED SHAPE = {proc_t1.shape}", fontsize=18)
plt.axis("off")

# Mask WT
plt.subplot(3,3,7)
show_slice(WT.astype(int), "WT (Whole Tumor)", cmap="hot")

# Mask TC
plt.subplot(3,3,8)
show_slice(TC.astype(int), "TC (Tumor Core)", cmap="hot")

# Mask ET
plt.subplot(3,3,9)
show_slice(ET.astype(int), "ET (Enhancing Tumor)", cmap="hot")

plt.tight_layout()
plt.show()
