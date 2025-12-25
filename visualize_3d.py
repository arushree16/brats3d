#!/usr/bin/env python3
"""
3D visualization script using PyVista for BraTS segmentation results
Creates publication-ready 3D renders of tumor segmentations
"""

import sys
import numpy as np
import torch
import pyvista as pv
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from unet3d import UNet3D

class BraTS3DVisualizer:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.colors = {
            'tumor_core': 'red',
            'edema': 'green',
            'enhancing': 'magenta',
            'whole_tumor': 'lightblue'
        }
        
    def load_model(self, checkpoint_path, attention_type='none'):
        """Load trained model from checkpoint"""
        model = UNet3D(attention_type=attention_type)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
    
    def preprocess_mask(self, mask):
        """Convert BraTS labels to 3-class format"""
        mask_np = mask.clone()
        mask_np[mask == 4] = 1  # enhancing tumor -> tumor core
        mask_np[mask_np > 2] = 2  # safety: any remaining values > 2 -> edema
        return mask_np.long()
    
    def predict_batch(self, model, images):
        """Get model predictions"""
        with torch.no_grad():
            images = images.to(self.device, dtype=torch.float32)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
        return preds.cpu().numpy()
    
    def mask_to_mesh(self, mask, label, spacing=(1.0, 1.0, 1.0)):
        """Convert binary mask to mesh using PyVista"""
        # Create grid
        grid = pv.UniformGrid()
        grid.dimensions = np.array(mask.shape) + 1
        grid.spacing = spacing
        grid.origin = (0, 0, 0)
        
        # Add mask data
        grid.cell_data["values"] = mask.flatten(order="F")
        
        # Extract surface
        mesh = grid.contour([0.5])  # Extract isosurface at 0.5
        return mesh
    
    def create_brats_regions(self, mask_3class):
        """Create meshes for different BraTS regions from 3-class mask"""
        regions = {}
        
        # Tumor Core (label 1)
        tc_mask = (mask_3class == 1).astype(np.uint8)
        if tc_mask.sum() > 0:
            regions['tumor_core'] = self.mask_to_mesh(tc_mask, 1)
        
        # Edema (label 2)  
        edema_mask = (mask_3class == 2).astype(np.uint8)
        if edema_mask.sum() > 0:
            regions['edema'] = self.mask_to_mesh(edema_mask, 2)
        
        # Whole Tumor (labels 1 + 2)
        wt_mask = ((mask_3class == 1) | (mask_3class == 2)).astype(np.uint8)
        if wt_mask.sum() > 0:
            regions['whole_tumor'] = self.mask_to_mesh(wt_mask, 3)
        
        return regions
    
    def create_3d_visualization(self, image, gt_mask, pred_mask, 
                               model_name="Model", save_path=None):
        """Create 3D visualization comparing ground truth and prediction"""
        plotter = pv.Plotter(off_screen=True if save_path else False)
        
        # Add brain volume as background (use T1 modality if available)
        if len(image.shape) == 4:
            brain_vol = image[..., 0]  # Use first modality
        else:
            brain_vol = image
            
        # Create brain mesh for context
        brain_grid = pv.UniformGrid()
        brain_grid.dimensions = np.array(brain_vol.shape) + 1
        brain_grid.spacing = (1.0, 1.0, 1.0)
        brain_grid.cell_data["values"] = brain_vol.flatten(order="F")
        
        # Add brain volume with low opacity
        plotter.add_volume(brain_grid, cmap="gray", opacity=0.1, clim=[brain_vol.min(), brain_vol.max()])
        
        # Create and add ground truth regions (left side)
        gt_regions = self.create_brats_regions(gt_mask)
        for region_name, mesh in gt_regions.items():
            if mesh.n_points > 0:
                plotter.add_mesh(
                    mesh, 
                    color=self.colors[region_name],
                    opacity=0.6,
                    name=f"GT_{region_name}",
                    style='surface'
                )
        
        # Create and add prediction regions (right side - shifted)
        pred_regions = self.create_brats_regions(pred_mask)
        shift_x = brain_vol.shape[0] + 20  # Shift prediction to the right
        
        for region_name, mesh in pred_regions.items():
            if mesh.n_points > 0:
                shifted_mesh = mesh.translate((shift_x, 0, 0))
                plotter.add_mesh(
                    shifted_mesh,
                    color=self.colors[region_name],
                    opacity=0.8,
                    name=f"Pred_{region_name}",
                    style='surface'
                )
        
        # Set up the view
        plotter.add_text("Ground Truth", position=(10, 10), font_size=12)
        plotter.add_text("Prediction", position=(shift_x, 10), font_size=12)
        plotter.add_text(model_name, position=(10, brain_vol.shape[1] - 20), font_size=14, font='bold')
        
        # Set camera and lighting
        plotter.camera_position = 'iso'
        plotter.enable_shadows()
        
        # Save or show
        if save_path:
            plotter.screenshot(save_path)
            plotter.close()
        else:
            plotter.show()
        
        return plotter
    
    def create_comparison_3d(self, image, gt_mask, pred_masks, model_names, save_path=None):
        """Create 3D comparison of multiple models"""
        n_models = len(pred_masks)
        plotter = pv.Plotter(off_screen=True if save_path else False, shape=(1, n_models))
        
        # Add brain volume to each subplot
        if len(image.shape) == 4:
            brain_vol = image[..., 0]
        else:
            brain_vol = image
            
        brain_grid = pv.UniformGrid()
        brain_grid.dimensions = np.array(brain_vol.shape) + 1
        brain_grid.spacing = (1.0, 1.0, 1.0)
        brain_grid.cell_data["values"] = brain_vol.flatten(order="F")
        
        # Ground truth in first subplot
        plotter.subplot(0, 0)
        plotter.add_volume(brain_grid, cmap="gray", opacity=0.1)
        
        gt_regions = self.create_brats_regions(gt_mask)
        for region_name, mesh in gt_regions.items():
            if mesh.n_points > 0:
                plotter.add_mesh(mesh, color=self.colors[region_name], opacity=0.6)
        plotter.add_text("Ground Truth", position=(10, 10))
        
        # Predictions in remaining subplots
        for i, (pred_mask, model_name) in enumerate(zip(pred_masks, model_names), 1):
            plotter.subplot(0, i)
            plotter.add_volume(brain_grid, cmap="gray", opacity=0.1)
            
            pred_regions = self.create_brats_regions(pred_mask)
            for region_name, mesh in pred_regions.items():
                if mesh.n_points > 0:
                    plotter.add_mesh(mesh, color=self.colors[region_name], opacity=0.8)
            plotter.add_text(model_name, position=(10, 10))
        
        # Link cameras for synchronized rotation
        plotter.link_views()
        
        if save_path:
            plotter.screenshot(save_path)
            plotter.close()
        else:
            plotter.show()
        
        return plotter
    
    def create_rotating_animation(self, image, gt_mask, pred_mask, model_name="Model", 
                                save_path=None, n_frames=36):
        """Create rotating 3D animation"""
        plotter = pv.Plotter(off_screen=True)
        
        # Add brain volume
        if len(image.shape) == 4:
            brain_vol = image[..., 0]
        else:
            brain_vol = image
            
        brain_grid = pv.UniformGrid()
        brain_grid.dimensions = np.array(brain_vol.shape) + 1
        brain_grid.spacing = (1.0, 1.0, 1.0)
        brain_grid.cell_data["values"] = brain_vol.flatten(order="F")
        plotter.add_volume(brain_grid, cmap="gray", opacity=0.1)
        
        # Add segmentation
        gt_regions = self.create_brats_regions(gt_mask)
        for region_name, mesh in gt_regions.items():
            if mesh.n_points > 0:
                plotter.add_mesh(mesh, color=self.colors[region_name], opacity=0.6)
        
        pred_regions = self.create_brats_regions(pred_mask)
        shift_x = brain_vol.shape[0] + 20
        for region_name, mesh in pred_regions.items():
            if mesh.n_points > 0:
                shifted_mesh = mesh.translate((shift_x, 0, 0))
                plotter.add_mesh(shifted_mesh, color=self.colors[region_name], opacity=0.8)
        
        plotter.add_text(model_name, position=(10, brain_vol.shape[1] - 20), font_size=14, font='bold')
        plotter.camera_position = 'iso'
        
        # Create animation frames
        if save_path:
            plotter.open_gif(save_path)
            for i in range(n_frames):
                plotter.camera.azimuth += 360 / n_frames
                plotter.write_frame()
            plotter.close()
        else:
            plotter.show()

def main():
    """Main 3D visualization pipeline"""
    visualizer = BraTS3DVisualizer()
    
    # Example usage - modify paths as needed
    output_dir = Path("outputs/3d_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models (example paths)
    models = {
        'Baseline': visualizer.load_model("outputs/baseline/best.pth", 'none'),
        'SE-UNet': visualizer.load_model("outputs/se/best.pth", 'se'),
        'CBAM-UNet': visualizer.load_model("outputs/cbam/best.pth", 'cbam'),
        'Hybrid': visualizer.load_model("outputs/hybrid/best.pth", 'hybrid')
    }
    
    print("Generating 3D visualizations...")
    
    # Load sample data (placeholder - replace with actual data loading)
    sample_image = np.random.rand(128, 128, 128, 4)
    sample_gt = np.random.randint(0, 3, (128, 128, 128))
    
    # Get predictions
    pred_masks = []
    model_names = []
    
    for name, model in models.items():
        image_tensor = torch.from_numpy(sample_image).permute(3, 0, 1, 2).unsqueeze(0)
        pred = visualizer.predict_batch(model, image_tensor)
        pred_masks.append(pred[0])
        model_names.append(name)
    
    # Generate individual 3D visualizations
    for pred_mask, model_name in zip(pred_masks, model_names):
        save_path = output_dir / f"{model_name.lower()}_3d.png"
        visualizer.create_3d_visualization(
            sample_image, sample_gt, pred_mask, model_name, save_path
        )
        print(f"Saved 3D visualization for {model_name}")
    
    # Generate comparison 3D visualization
    save_path = output_dir / "all_models_3d_comparison.png"
    visualizer.create_comparison_3d(
        sample_image, sample_gt, pred_masks, model_names, save_path
    )
    print(f"Saved 3D comparison visualization")
    
    # Generate rotating animation for best model
    best_model_idx = 0  # Assuming first model is best
    save_path = output_dir / "best_model_3d_animation.gif"
    visualizer.create_rotating_animation(
        sample_image, sample_gt, pred_masks[best_model_idx], 
        model_names[best_model_idx], save_path
    )
    print(f"Saved 3D animation")
    
    print(f"\n3D visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()
