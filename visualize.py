#!/usr/bin/env python3
"""
Visualization script for BraTS segmentation results
Creates paper-ready figures comparing baseline vs attention models
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import nibabel as nib
from tqdm import tqdm
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
            'background': [0, 0, 0],
            'tumor_core': [1, 0, 0],      # Red
            'edema': [0, 1, 0],          # Green  
            'enhancing': [1, 0, 1],      # Magenta
            'whole_tumor': [0.5, 0.5, 1] # Light blue
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
    
    def create_overlay(self, image, gt_mask, pred_mask, slice_idx, axis='axial'):
        """Create overlay visualization of image with masks"""
        # Get slice based on axis
        if axis == 'axial':
            img_slice = image[:, :, slice_idx]
            gt_slice = gt_mask[:, :, slice_idx]
            pred_slice = pred_mask[:, :, slice_idx]
        elif axis == 'coronal':
            img_slice = image[:, slice_idx, :]
            gt_slice = gt_mask[:, slice_idx, :]
            pred_slice = pred_mask[:, slice_idx, :]
        else:  # sagittal
            img_slice = image[slice_idx, :, :]
            gt_slice = gt_mask[slice_idx, :, :]
            pred_slice = pred_mask[slice_idx, :, :]
        
        # Normalize image slice
        img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
        
        # Create RGB image
        rgb_img = np.stack([img_slice] * 3, axis=-1)
        
        # Add ground truth overlay (semi-transparent)
        gt_overlay = self.create_mask_overlay(gt_slice, alpha=0.3)
        pred_overlay = self.create_mask_overlay(pred_slice, alpha=0.5)
        
        return rgb_img, gt_overlay, pred_overlay
    
    def create_mask_overlay(self, mask_slice, alpha=0.5):
        """Create colored overlay for mask slice"""
        overlay = np.zeros((*mask_slice.shape, 3))
        
        # Apply colors for each class
        overlay[mask_slice == 1] = self.colors['tumor_core']  # Tumor core
        overlay[mask_slice == 2] = self.colors['edema']        # Edema
        
        return overlay * alpha
    
    def plot_comparison(self, image, gt_mask, pred_mask, slice_idx, 
                       model_name="Model", save_path=None):
        """Create multi-planar comparison plot"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{model_name} - Slice {slice_idx}', fontsize=16, fontweight='bold')
        
        # Axial view
        for col, view in enumerate(['axial', 'coronal', 'sagittal']):
            rgb_img, gt_overlay, pred_overlay = self.create_overlay(
                image, gt_mask, pred_mask, slice_idx, view
            )
            
            # Ground truth
            axes[0, col].imshow(rgb_img)
            axes[0, col].imshow(gt_overlay)
            axes[0, col].set_title(f'Ground Truth - {view.capitalize()}')
            axes[0, col].axis('off')
            
            # Prediction
            axes[1, col].imshow(rgb_img)
            axes[1, col].imshow(pred_overlay)
            axes[1, col].set_title(f'Prediction - {view.capitalize()}')
            axes[1, col].axis('off')
        
        # Add legend
        legend_elements = [
            patches.Patch(color=self.colors['tumor_core'], label='Tumor Core'),
            patches.Patch(color=self.colors['edema'], label='Edema'),
        ]
        axes[0, 0].legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def create_segmentation_comparison(self, image, gt_mask, pred_masks, 
                                     model_names, slice_idx, save_path=None):
        """Compare multiple models side by side"""
        n_models = len(pred_masks)
        fig, axes = plt.subplots(n_models + 1, 3, figsize=(15, 5 * (n_models + 1)))
        fig.suptitle(f'Segmentation Comparison - Slice {slice_idx}', 
                    fontsize=16, fontweight='bold')
        
        views = ['axial', 'coronal', 'sagittal']
        
        # Ground truth row
        for col, view in enumerate(views):
            rgb_img, gt_overlay, _ = self.create_overlay(
                image, gt_mask, gt_mask, slice_idx, view
            )
            axes[0, col].imshow(rgb_img)
            axes[0, col].imshow(gt_overlay)
            axes[0, col].set_title(f'Ground Truth - {view.capitalize()}')
            axes[0, col].axis('off')
        
        # Model predictions
        for row, (pred_mask, model_name) in enumerate(zip(pred_masks, model_names), 1):
            for col, view in enumerate(views):
                rgb_img, _, pred_overlay = self.create_overlay(
                    image, gt_mask, pred_mask, slice_idx, view
                )
                axes[row, col].imshow(rgb_img)
                axes[row, col].imshow(pred_overlay)
                axes[row, col].set_title(f'{model_name} - {view.capitalize()}')
                axes[row, col].axis('off')
        
        # Add legend
        legend_elements = [
            patches.Patch(color=self.colors['tumor_core'], label='Tumor Core'),
            patches.Patch(color=self.colors['edema'], label='Edema'),
        ]
        axes[0, 0].legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def compute_metrics_table(self, gt_mask, pred_masks, model_names):
        """Compute metrics for comparison table"""
        results = []
        
        for pred_mask, model_name in zip(pred_masks, model_names):
            # Convert to tensors
            gt_tensor = torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0)
            pred_tensor = torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0)
            
            # Compute BraTS metrics
            metrics = compute_brats_regions_metrics(pred_tensor, gt_tensor)
            
            results.append({
                'Model': model_name,
                'WT Dice': f"{metrics['wt_dice']:.3f}",
                'TC Dice': f"{metrics['tc_dice']:.3f}",
                'WT HD95': f"{metrics['wt_hd95']:.1f}",
                'TC HD95': f"{metrics['tc_hd95']:.1f}"
            })
        
        return results

def main():
    """Main visualization pipeline"""
    visualizer = BraTSVisualizer()
    
    # Example usage - modify paths as needed
    data_path = "data/processed/brats128"
    output_dir = Path("outputs/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models (example paths)
    models = {
        'Baseline': visualizer.load_model("outputs/baseline/best.pth", 'none'),
        'SE-UNet': visualizer.load_model("outputs/se/best.pth", 'se'),
        'CBAM-UNet': visualizer.load_model("outputs/cbam/best.pth", 'cbam'),
        'Hybrid': visualizer.load_model("outputs/hybrid/best.pth", 'hybrid')
    }
    
    print("Generating visualizations...")
    
    # Load sample data (you'll need to adapt this to your data loading)
    # This is a placeholder - replace with actual data loading
    sample_image = np.random.rand(128, 128, 128, 4)  # Example shape
    sample_gt = np.random.randint(0, 3, (128, 128, 128))  # Example mask
    
    # Get predictions from all models
    pred_masks = []
    model_names = []
    
    for name, model in models.items():
        # Convert to tensor and predict
        image_tensor = torch.from_numpy(sample_image).permute(3, 0, 1, 2).unsqueeze(0)
        pred = visualizer.predict_batch(model, image_tensor)
        pred_masks.append(pred[0])  # Remove batch dimension
        model_names.append(name)
    
    # Generate comparison visualizations
    slice_idx = 64  # Middle slice
    
    # Individual model comparisons
    for pred_mask, model_name in zip(pred_masks, model_names):
        fig = visualizer.plot_comparison(
            sample_image[:, :, :, 0],  # Use first modality
            sample_gt, pred_mask, slice_idx, model_name,
            save_path=output_dir / f"{model_name.lower()}_comparison.png"
        )
        plt.close(fig)
    
    # Multi-model comparison
    fig = visualizer.create_segmentation_comparison(
        sample_image[:, :, :, 0], sample_gt, pred_masks, 
        model_names, slice_idx, 
        save_path=output_dir / "all_models_comparison.png"
    )
    plt.close(fig)
    
    # Metrics table
    metrics_table = visualizer.compute_metrics_table(sample_gt, pred_masks, model_names)
    
    print("\nMetrics Comparison:")
    print("-" * 80)
    for result in metrics_table:
        print(f"{result['Model']:10} | WT: {result['WT Dice']} ({result['WT HD95']}) | "
              f"TC: {result['TC Dice']} ({result['TC HD95']})")
    
    print(f"\nVisualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()
