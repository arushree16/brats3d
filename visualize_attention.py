#!/usr/bin/env python3
"""
Attention Map Visualization for 3D U-Net Models
Visualizes what SE, CBAM, and Hybrid attention mechanisms focus on
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm

# Ensure src is importable
import sys
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from unet3d import UNet3D
from dataset_torchio import make_loaders
from attention import SEBlock3D, CBAM3D

class AttentionHook:
    """Hook to capture attention maps during forward pass"""
    def __init__(self):
        self.attention_maps = []
        self.layer_names = []
        
    def hook_fn(self, module, input, output, layer_name):
        """Store attention maps"""
        if hasattr(module, 'attention_weights'):
            self.attention_maps.append(module.attention_weights.clone())
            self.layer_names.append(layer_name)
        elif hasattr(module, 'channel_weights'):
            self.attention_maps.append(module.channel_weights.clone())
            self.layer_names.append(layer_name)

def create_attention_model(model_type):
    """Create model with attention hooks"""
    model = UNet3D(in_channels=4, base_filters=32, num_classes=3, attention_type=model_type)
    
    hooks = []
    hook_manager = AttentionHook()
    
    # Add hooks to attention blocks
    for name, module in model.named_modules():
        if 'se' in name.lower() or 'cbam' in name.lower():
            if hasattr(module, 'se') or hasattr(module, 'cbam'):
                hook = module.register_forward_hook(
                    lambda m, i, o, n=name: hook_manager.hook_fn(m, i, o, n)
                )
                hooks.append(hook)
    
    return model, hooks, hook_manager

def extract_attention_weights(model, x, layer_name):
    """Extract attention weights from specific layer"""
    # This is a simplified version - you'll need to modify your attention blocks
    # to store attention weights during forward pass
    
    # For SE blocks - channel attention weights
    # For CBAM blocks - channel + spatial attention weights
    
    # We'll need to modify your attention.py to expose these weights
    pass

def visualize_attention_comparison():
    """Create attention map comparison visualization"""
    
    print("🔍 Creating Attention Map Visualizations...")
    
    # Load validation data
    _, val_loader = make_loaders('data/processed/brats128', batch_size=1, augment=False)
    
    # Load trained models
    models = {}
    model_types = ['baseline', 'se', 'cbam', 'hybrid', 'se_encoder_only', 'cbam_bottleneck_only']
    
    for model_type in model_types:
        try:
            model_path = f'final results/{model_type}/best.pth'
            if Path(model_path).exists():
                model = UNet3D(in_channels=4, base_filters=32, num_classes=3, attention_type=model_type)
                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                models[model_type] = model
                print(f"✅ Loaded {model_type} model")
            else:
                print(f"❌ {model_type} model not found at {model_path}")
        except Exception as e:
            print(f"❌ Error loading {model_type}: {e}")
    
    if len(models) < 2:
        print("❌ Need at least 2 models for comparison")
        return
    
    # Get sample data
    sample_batch = next(iter(val_loader))
    sample_image, sample_mask = sample_batch
    
    print(f"📊 Sample image shape: {sample_image.shape}")
    
    # Create attention visualization figure
    fig, axes = plt.subplots(6, 5, figsize=(20, 24))
    fig.suptitle('Attention Map Comparison: What Do Different Models Focus On?', 
                 fontsize=16, fontweight='bold')
    
    # Select a middle slice for visualization
    slice_idx = sample_image.shape[2] // 2  # Middle depth slice
    
    # Original image and ground truth
    axes[0, 0].imshow(sample_image[0, 0, slice_idx].cpu().numpy(), cmap='gray')
    axes[0, 0].set_title('Original MRI (T1)', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(sample_image[0, 1, slice_idx].cpu().numpy(), cmap='gray')
    axes[0, 1].set_title('Original MRI (T1ce)', fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(sample_image[0, 2, slice_idx].cpu().numpy(), cmap='gray')
    axes[0, 2].set_title('Original MRI (T2)', fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(sample_image[0, 3, slice_idx].cpu().numpy(), cmap='gray')
    axes[0, 3].set_title('Original MRI (FLAIR)', fontweight='bold')
    axes[0, 3].axis('off')
    
    axes[0, 4].imshow(sample_mask[0, slice_idx].cpu().numpy(), cmap='jet')
    axes[0, 4].set_title('Ground Truth', fontweight='bold')
    axes[0, 4].axis('off')
    
    # Row labels
    row_labels = ['Input', 'Baseline', 'SE-UNet', 'CBAM-UNet', 'Hybrid', 'SE-Encoder Only', 'CBAM-Bottleneck Only']
    
    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label, rotation=90, fontsize=12, fontweight='bold')
    
    # Visualize attention for each model
    model_colors = {
        'baseline': 'blue', 
        'se': 'orange', 
        'cbam': 'green', 
        'hybrid': 'red',
        'se_encoder_only': 'purple',
        'cbam_bottleneck_only': 'brown'
    }
    labels = {
        'baseline': 'Baseline', 
        'se': 'SE-UNet', 
        'cbam': 'CBAM-UNet', 
        'hybrid': 'Hybrid',
        'se_encoder_only': 'SE-Encoder Only',
        'cbam_bottleneck_only': 'CBAM-Bottleneck Only'
    }
    
    for model_type, config in labels.items():
        if model_type not in models:
            # Skip if model not loaded
            for col in range(5):
                axes[config['row'], col].text(0.5, 0.5, 'Model Not Available', 
                                          ha='center', va='center', 
                                          transform=axes[config['row'], col].transAxes)
                axes[config['row'], col].axis('off')
            continue
        
        row = config['row']
        model = models[model_type]
        
        with torch.no_grad():
            # Get model prediction
            output = model(sample_image)
            pred = torch.argmax(output, dim=1)[0]
            
            # Show prediction
            axes[row, 0].imshow(pred[slice_idx].cpu().numpy(), cmap='jet')
            axes[row, 0].set_title(f'{config["title"]} - Prediction', fontweight='bold')
            axes[row, 0].axis('off')
            
            # TODO: Extract and visualize actual attention maps
            # This requires modifying your attention blocks to store weights
            
            # Placeholder for attention visualizations
            for col in range(1, 5):
                if model_type == 'baseline':
                    # No attention - show gradient-based saliency instead
                    axes[row, col].text(0.5, 0.5, 'No Attention\n(Gradient Saliency)', 
                                       ha='center', va='center', 
                                       transform=axes[row, col].transAxes)
                else:
                    axes[row, col].text(0.5, 0.5, f'{model_type.upper()}\nAttention Map', 
                                       ha='center', va='center', 
                                       transform=axes[row, col].transAxes)
                axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(6):
        for j in range(5):
            if i >= len(row_labels) or (i == 0 and j >= 2):
                axes[i, j].axis('off')
            else:
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/attention_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Attention comparison saved to results/attention_comparison.png")

def create_attention_flowchart():
    """Create flowchart showing attention mechanism differences"""
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Attention Mechanism Comparison: How They Work', fontsize=16, fontweight='bold')
    
    # Baseline - No attention
    ax = axes[0]
    ax.text(0.5, 0.8, 'Input Features', ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax.text(0.5, 0.5, '3D Convolution', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax.text(0.5, 0.2, 'Output Features', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    ax.text(0.5, 0.05, 'No Attention', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='red')
    ax.set_title('Baseline', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # SE - Channel attention
    ax = axes[1]
    ax.text(0.5, 0.8, 'Input Features', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax.text(0.5, 0.6, 'Global Avg Pool', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    ax.text(0.5, 0.4, 'MLP (Channel Weights)', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax.text(0.5, 0.2, 'Channel Recalibration', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    ax.text(0.5, 0.05, 'Channel Attention', ha='center', va='center',
            fontsize=12, fontweight='bold', color='orange')
    ax.set_title('SE-UNet', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # CBAM - Channel + Spatial attention
    ax = axes[2]
    ax.text(0.5, 0.8, 'Input Features', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax.text(0.5, 0.6, 'Channel Attention', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    ax.text(0.5, 0.4, 'Spatial Attention', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax.text(0.5, 0.2, 'Channel × Spatial', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    ax.text(0.5, 0.05, 'CBAM Attention', ha='center', va='center',
            fontsize=12, fontweight='bold', color='green')
    ax.set_title('CBAM-UNet', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Hybrid - Strategic placement
    ax = axes[3]
    ax.text(0.5, 0.8, 'Encoder: SE', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax.text(0.5, 0.6, 'Bottleneck: CBAM', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    ax.text(0.5, 0.4, 'Decoder: None', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax.text(0.5, 0.2, 'Strategic Placement', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    ax.text(0.5, 0.05, 'Hybrid Attention', ha='center', va='center',
            fontsize=12, fontweight='bold', color='red')
    ax.set_title('Hybrid', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/attention_mechanism_flowchart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Attention mechanism flowchart saved to results/attention_mechanism_flowchart.png")

if __name__ == "__main__":
    from pathlib import Path
    Path("results").mkdir(exist_ok=True)
    
    print("🔍 Attention Map Visualization for 3D U-Net Models")
    print("=" * 60)
    
    # Create attention mechanism flowchart
    create_attention_flowchart()
    
    # Create attention comparison (placeholder - needs attention weight extraction)
    visualize_attention_comparison()
    
    print("\n📊 Attention Visualizations Created:")
    print("-" * 40)
    print("1. attention_mechanism_flowchart.png - How each attention works")
    print("2. attention_comparison.png - What each model focuses on")
    print("\n🎯 Paper Usage:")
    print("- Figure 10: Attention mechanism comparison")
    print("- Figure 11: Attention map visualization")
    print("- Answers: 'Why does placement matter?'")
