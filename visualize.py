#!/usr/bin/env python3
"""
Publication-ready visualization script for BraTS segmentation
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from unet3d import UNet3D
from metrics import compute_brats_regions_metrics


class BraTSVisualizer:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.colors = {
            'tumor_core': [1, 0, 0],   # red
            'edema': [0, 1, 0],        # green
        }

    # ---------------- MODEL ----------------
    def load_model(self, checkpoint_path, attention_type='none'):
        model = UNet3D(attention_type=attention_type, base_filters=16)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model

    # ---------------- UTILITIES ----------------
    def find_best_slice(self, mask):
        """Find slice with max tumor"""
        slice_sums = np.sum(mask > 0, axis=(0, 1))
        return np.argmax(slice_sums)

    def dice_score(self, pred, gt, cls=1):
        pred_bin = (pred == cls)
        gt_bin = (gt == cls)

        intersection = np.sum(pred_bin * gt_bin)
        union = np.sum(pred_bin) + np.sum(gt_bin)

        if union == 0:
            return 1.0
        return 2.0 * intersection / union

    def create_overlay(self, image, gt_mask, pred_mask, slice_idx):
        img_slice = image[:, :, slice_idx]
        gt_slice = gt_mask[:, :, slice_idx]
        pred_slice = pred_mask[:, :, slice_idx]

        img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)

        rgb_img = np.stack([img_slice] * 3, axis=-1)

        gt_overlay = np.zeros_like(rgb_img)
        pred_overlay = np.zeros_like(rgb_img)

        gt_overlay[gt_slice == 1] = self.colors['tumor_core']
        gt_overlay[gt_slice == 2] = self.colors['edema']

        pred_overlay[pred_slice == 1] = self.colors['tumor_core']
        pred_overlay[pred_slice == 2] = self.colors['edema']

        return rgb_img, gt_overlay * 0.3, pred_overlay * 0.5

    # ---------------- PAPER FIGURE ----------------
    def create_paper_figure(self, image, gt_mask, pred_masks, model_names, slice_idx, save_path=None):
        n_models = len(pred_masks)
        fig, axes = plt.subplots(1, n_models + 2, figsize=(4 * (n_models + 2), 4))

        img_slice = image[:, :, slice_idx]
        img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)

        # MRI
        axes[0].imshow(img_slice, cmap='gray')
        axes[0].set_title("MRI", fontsize=12)
        axes[0].axis('off')

        # Ground truth
        _, gt_overlay, _ = self.create_overlay(image, gt_mask, gt_mask, slice_idx)
        axes[1].imshow(img_slice, cmap='gray')
        axes[1].imshow(gt_overlay)
        axes[1].set_title("Ground Truth", fontsize=12)
        axes[1].axis('off')

        # Predictions
        for i, (pred_mask, name) in enumerate(zip(pred_masks, model_names)):
            _, _, pred_overlay = self.create_overlay(image, gt_mask, pred_mask, slice_idx)

            dice = self.dice_score(pred_mask, gt_mask)

            axes[i + 2].imshow(img_slice, cmap='gray')
            axes[i + 2].imshow(pred_overlay)
            axes[i + 2].set_title(f"{name}\nDice: {dice:.3f}", fontsize=11)
            axes[i + 2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    # ---------------- METRICS ----------------
    def compute_metrics_table(self, gt_mask, pred_masks, model_names):
        results = []

        for pred_mask, model_name in zip(pred_masks, model_names):
            gt_tensor = torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0)
            pred_tensor = torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0)

            metrics = compute_brats_regions_metrics(pred_tensor, gt_tensor)

            results.append({
                'Model': model_name,
                'WT Dice': f"{metrics['wt_dice']:.3f}",
                'TC Dice': f"{metrics['tc_dice']:.3f}",
            })

        return results


# ================= MAIN =================
def main():
    visualizer = BraTSVisualizer()

    data_path = "data/processed/brats128"
    output_dir = Path("outputs/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    models = {
        'Baseline': visualizer.load_model("/content/drive/MyDrive/brain_tumor_logs/baseline/best.pth", 'none'),
        'SE': visualizer.load_model("/content/drive/MyDrive/brain_tumor_logs/se/best.pth", 'se'),
        'CBAM': visualizer.load_model("/content/drive/MyDrive/brain_tumor_logs/cbam/best.pth", 'cbam'),
        'Hybrid': visualizer.load_model("/content/drive/MyDrive/brain_tumor_logs/hybrid/best.pth", 'hybrid'),
        'SE-Enc': visualizer.load_model("/content/drive/MyDrive/brain_tumor_logs/se_encoder_only/best.pth", 'se_encoder_only'),
        'CBAM-Bottleneck': visualizer.load_model("/content/drive/MyDrive/brain_tumor_logs/cbam_bottleneck_only/best.pth", 'cbam_bottleneck_only')
    }

    print("Generating visualizations...")

    from dataset_torchio import make_loaders
    _, val_loader = make_loaders(data_path, batch_size=1, num_workers=0, shuffle_train=False, augment=False)

    sample_batch = next(iter(val_loader))
    sample_image = sample_batch[0].numpy()   # KEEP SHAPE [1, 4, 128, 128, 128]
    sample_gt = sample_batch[1].numpy()

    if sample_gt.shape[0] == 1:
        sample_gt = sample_gt[0]

    print(f"📊 Image shape: {sample_image.shape}")
    print(f"📊 GT shape: {sample_gt.shape}")

    # Best slice
    slice_idx = visualizer.find_best_slice(sample_gt)
    print(f"📍 Best slice: {slice_idx}")

    # Predictions
    pred_masks = []
    model_names = []

    for name, model in models.items():
        print(f"Processing {name}...")

        image_tensor = torch.from_numpy(sample_image).to(visualizer.device, dtype=torch.float32)

        with torch.no_grad():
            pred = model(image_tensor)
            pred_mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

        pred_masks.append(pred_mask)
        model_names.append(name)

    # Paper figure
    fig = visualizer.create_paper_figure(
        sample_image[0][0],  # first modality
        sample_gt,
        pred_masks,
        model_names,
        slice_idx,
        save_path=output_dir / "paper_comparison.png"
    )
    plt.close(fig)

    # Metrics
    metrics = visualizer.compute_metrics_table(sample_gt, pred_masks, model_names)

    print("\nMetrics:")
    for m in metrics:
        print(m)


if __name__ == "__main__":
    main()