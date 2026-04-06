#!/usr/bin/env python3
"""
Fixed + Publication-ready 3D visualization for BraTS
"""

import sys
import numpy as np
import torch
import pyvista as pv
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ✅ CRITICAL FIX for Colab
pv.start_xvfb()

# Add src
sys.path.append(str(Path(__file__).resolve().parent / "src"))
from unet3d import UNet3D


class BraTS3DVisualizer:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.colors = {
            'tumor_core': 'red',
            'edema': 'green',
            'whole_tumor': 'lightblue'
        }

    # ---------------- MODEL ----------------
    def load_model(self, path, attention_type):
        model = UNet3D(attention_type=attention_type, base_filters=16)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model

    # ---------------- FIXED MESH ----------------
    def mask_to_mesh(self, mask):
        """Convert mask → mesh (FIXED)"""

        grid = pv.ImageData()
        grid.dimensions = np.array(mask.shape) + 1
        grid.spacing = (1, 1, 1)

        # assign as CELL data
        grid.cell_data["values"] = mask.flatten(order="F")

        # 🔥 FIX: convert to POINT DATA
        grid = grid.cell_data_to_point_data()

        mesh = grid.contour([0.5])
        return mesh

    def create_regions(self, mask):
        regions = {}

        tc = (mask == 1).astype(np.uint8)
        ed = (mask == 2).astype(np.uint8)
        wt = ((mask == 1) | (mask == 2)).astype(np.uint8)

        if tc.sum() > 0:
            regions['tumor_core'] = self.mask_to_mesh(tc)
        if ed.sum() > 0:
            regions['edema'] = self.mask_to_mesh(ed)
        if wt.sum() > 0:
            regions['whole_tumor'] = self.mask_to_mesh(wt)

        return regions

    # ---------------- VISUALIZATION ----------------
    def create_3d_visualization(self, image, gt, pred, name, save_path):
        plotter = pv.Plotter(off_screen=True)

        # Use first modality
        if image.ndim == 4:
            image = image[0]

        # Brain volume
        grid = pv.ImageData()
        grid.dimensions = np.array(image.shape) + 1
        grid.cell_data["values"] = image.flatten(order="F")

        plotter.add_volume(grid, cmap="gray", opacity=0.08)

        # Ground truth
        gt_regions = self.create_regions(gt)
        for k, mesh in gt_regions.items():
            plotter.add_mesh(mesh, color=self.colors[k], opacity=0.5)

        # Prediction (shifted)
        pred_regions = self.create_regions(pred)
        shift = image.shape[0] + 10

        for k, mesh in pred_regions.items():
            plotter.add_mesh(mesh.translate((shift, 0, 0)), color=self.colors[k], opacity=0.8)

        plotter.add_text(f"{name}", font_size=12)

        plotter.camera_position = 'iso'

        plotter.screenshot(save_path)
        plotter.close()

    # ---------------- MAIN ----------------
def main():
    vis = BraTS3DVisualizer()

    output_dir = Path("outputs/3d_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    models = {
        'Baseline': vis.load_model("/content/drive/MyDrive/brain_tumor_logs/baseline/best.pth", 'none'),
        'SE': vis.load_model("/content/drive/MyDrive/brain_tumor_logs/se/best.pth", 'se'),
        'CBAM': vis.load_model("/content/drive/MyDrive/brain_tumor_logs/cbam/best.pth", 'cbam'),
    }

    # Load REAL data
    from dataset_torchio import make_loaders
    _, val_loader = make_loaders("data/processed/brats128", batch_size=1)

    batch = next(iter(val_loader))
    image = batch[0].numpy()     # [1,4,D,H,W]
    gt = batch[1].numpy()[0]

    # Predictions
    preds = []
    names = []

    for name, model in models.items():
        x = torch.from_numpy(image).to(vis.device, dtype=torch.float32)

        with torch.no_grad():
            out = model(x)
            pred = torch.argmax(out, dim=1).squeeze().cpu().numpy()

        preds.append(pred)
        names.append(name)

    # Generate
    for pred, name in zip(preds, names):
        vis.create_3d_visualization(
            image[0],
            gt,
            pred,
            name,
            output_dir / f"{name}_3d.png"
        )

    print("✅ 3D visualizations generated!")


if __name__ == "__main__":
    main()