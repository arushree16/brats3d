#!/usr/bin/env python3
"""
Modify attention blocks to store weights for visualization
This updates your attention.py to expose attention weights
"""

import torch
import torch.nn as nn

def create_enhanced_attention_blocks():
    """Create enhanced attention blocks that store weights for visualization"""
    
    enhanced_code = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
        # Store attention weights for visualization
        self.attention_weights = None

    def forward(self, x):
        b, c, _, _, _ = x.size()
        
        # Store attention weights for visualization
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        self.attention_weights = y.detach().clone()  # Store for visualization
        
        return x * y

class ChannelAttention3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        # Store attention weights for visualization
        self.channel_weights = None

    def forward(self, x):
        b, c, _, _, _ = x.size()
        
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_att = avg_out + max_out
        
        # Store channel attention weights for visualization
        self.channel_weights = torch.sigmoid(channel_att).detach().clone()
        
        return x * torch.sigmoid(channel_att).view(b, c, 1, 1, 1)

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # Store attention weights for visualization
        self.spatial_weights = None

    def forward(self, x):
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        
        # Generate spatial attention
        spatial_att = self.sigmoid(self.conv(combined))
        
        # Store spatial attention weights for visualization
        self.spatial_weights = spatial_att.detach().clone()
        
        return x * spatial_att

class CBAM3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention3D(channels, reduction)
        self.sa = SpatialAttention3D()
        
        # Store both attention weights for visualization
        self.channel_weights = None
        self.spatial_weights = None

    def forward(self, x):
        # Apply channel attention
        x = self.ca(x)
        self.channel_weights = self.ca.channel_weights.clone()
        
        # Apply spatial attention
        x = self.sa(x)
        self.spatial_weights = self.sa.spatial_weights.clone()
        
        return x
'''
    
    return enhanced_code

def update_attention_file():
    """Update the attention.py file with enhanced blocks"""
    
    enhanced_code = create_enhanced_attention_blocks()
    
    # Backup original file
    import shutil
    from pathlib import Path
    
    original_file = Path("src/attention.py")
    backup_file = Path("src/attention_backup.py")
    
    if original_file.exists():
        shutil.copy2(original_file, backup_file)
        print(f"✅ Backed up original attention.py to {backup_file}")
    
    # Write enhanced version
    with open(original_file, 'w') as f:
        f.write(enhanced_code)
    
    print("✅ Updated attention.py with visualization support")
    
    return True

def create_attention_extraction_script():
    """Create script to extract and visualize attention weights"""
    
    script_content = '''
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def extract_attention_maps(model, input_tensor, layer_name='conv_block'):
    """Extract attention maps from trained model"""
    
    attention_maps = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if hasattr(module, 'attention_weights'):
                attention_maps[f'{name}_se'] = module.attention_weights
            if hasattr(module, 'channel_weights'):
                attention_maps[f'{name}_channel'] = module.channel_weights
            if hasattr(module, 'spatial_weights'):
                attention_maps[f'{name}_spatial'] = module.spatial_weights
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if 'conv_block' in name.lower():
            if hasattr(module, 'se') or hasattr(module, 'cbam'):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return attention_maps

def visualize_attention_maps(attention_maps, save_path='results/attention_maps.png'):
    """Visualize extracted attention maps"""
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Attention Maps: What the Model Focuses On', fontsize=16, fontweight='bold')
    
    map_idx = 0
    for map_name, attention_data in attention_maps.items():
        if map_idx >= 8:  # Limit to 8 subplots
            break
            
        row = map_idx // 4
        col = map_idx % 4
        
        if 'se' in map_name or 'channel' in map_name:
            # Channel attention - visualize as bar chart
            weights = attention_data[0, :, 0, 0, 0].cpu().numpy()  # First sample, squeeze spatial dims
            axes[row, col].bar(range(len(weights)), weights)
            axes[row, col].set_title(f'{map_name}\\n(Channel Weights)')
            axes[row, col].set_xlabel('Channel Index')
            axes[row, col].set_ylabel('Weight')
            
        elif 'spatial' in map_name:
            # Spatial attention - visualize as heatmap
            spatial_map = attention_data[0, 0, :, :, :].cpu().numpy()  # First sample, first channel
            if len(spatial_map.shape) == 3:
                # Take middle slice of 3D spatial map
                mid_slice = spatial_map.shape[0] // 2
                im = axes[row, col].imshow(spatial_map[mid_slice], cmap='hot')
                axes[row, col].set_title(f'{map_name}\\n(Spatial Map - Slice {mid_slice})')
                plt.colorbar(im, ax=axes[row, col])
        
        map_idx += 1
    
    # Hide empty subplots
    for i in range(map_idx, 8):
        row = i // 4
        col = i % 4
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Attention maps saved to {save_path}")

def create_attention_comparison():
    """Create side-by-side comparison of attention mechanisms"""
    
    # This would load all 4 models and extract their attention maps
    # for the same input slice, showing what each focuses on
    
    print("🔍 Creating attention comparison visualization...")
    print("This requires:")
    print("1. Enhanced attention blocks (with weight storage)")
    print("2. All 4 trained models")
    print("3. Same input data for all models")
    
    return True
'''
    
    with open('extract_attention_maps.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Created attention extraction script")
    
    return True

if __name__ == "__main__":
    print("🔧 Enhancing Attention Blocks for Visualization")
    print("=" * 60)
    
    # Update attention.py with enhanced blocks
    update_attention_file()
    
    # Create attention extraction script
    create_attention_extraction_script()
    
    print("\\n🎯 Next Steps:")
    print("-" * 30)
    print("1. Retrain models with enhanced attention blocks")
    print("2. Use extract_attention_maps.py to visualize attention")
    print("3. Add attention visualizations to paper")
    print("\\n📊 This will create:")
    print("- Channel attention weight visualizations")
    print("- Spatial attention heatmaps")
    print("- Side-by-side model comparisons")
    print("- Direct evidence for 'why placement matters'")
