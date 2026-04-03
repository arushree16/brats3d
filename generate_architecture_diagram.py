#!/usr/bin/env python3
"""
Generate model architecture diagram for paper
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_architecture_diagram():
    """Create architecture diagram showing 4 model variants"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('3D U-Net Architecture Variants', fontsize=16, fontweight='bold')
    
    models = [
        ('Baseline', 'none', axes[0, 0]),
        ('SE-UNet', 'se', axes[0, 1]), 
        ('CBAM-UNet', 'cbam', axes[1, 0]),
        ('Hybrid', 'hybrid', axes[1, 1])
    ]
    
    colors = {
        'conv': 'lightblue',
        'attention': 'orange',
        'pool': 'gray',
        'upsample': 'lightgreen',
        'skip': 'red'
    }
    
    for (name, attn_type, ax) in models:
        ax.set_title(f'{name} ({attn_type.upper()})', fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Encoder path
        encoder_levels = [8, 6, 4, 2]
        for i, y in enumerate(encoder_levels):
            # Conv blocks
            conv = FancyBboxPatch((1, 0.8), (1, y), 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['conv'], 
                                edgecolor='black', linewidth=1)
            ax.add_patch(conv)
            ax.text(1.5, y+0.4, f'Conv\n{32*2**i}', 
                    ha='center', va='center', fontsize=8)
            
            # Attention blocks
            if attn_type == 'se' or (attn_type == 'hybrid' and i < 3):
                attn = FancyBboxPatch((1, 0.3), (3, y), 
                                    boxstyle="round,pad=0.05", 
                                    facecolor=colors['attention'], 
                                    edgecolor='black', linewidth=1)
                ax.add_patch(attn)
                ax.text(3.5, y+0.15, 'SE', 
                        ha='center', va='center', fontsize=8, fontweight='bold')
            elif attn_type == 'cbam':
                attn = FancyBboxPatch((1, 0.3), (3, y), 
                                    boxstyle="round,pad=0.05", 
                                    facecolor=colors['attention'], 
                                    edgecolor='black', linewidth=1)
                ax.add_patch(attn)
                ax.text(3.5, y+0.15, 'CBAM', 
                        ha='center', va='center', fontsize=8, fontweight='bold')
            elif attn_type == 'hybrid' and i == 3:  # Bottleneck
                attn = FancyBboxPatch((1, 0.3), (3, y), 
                                    boxstyle="round,pad=0.05", 
                                    facecolor=colors['attention'], 
                                    edgecolor='black', linewidth=1)
                ax.add_patch(attn)
                ax.text(3.5, y+0.15, 'CBAM', 
                        ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Pooling (except after bottleneck)
            if i < 3:
                pool = FancyBboxPatch((0.5, 0.4), (5, y), 
                                   boxstyle="round,pad=0.05", 
                                   facecolor=colors['pool'], 
                                   edgecolor='black', linewidth=1)
                ax.add_patch(pool)
                ax.text(5.25, y+0.2, 'MaxPool\n2x2x2', 
                        ha='center', va='center', fontsize=7)
                
                # Connections
                ax.plot([2, 2], [y+0.4, y-0.8], 'k-', linewidth=1)
                ax.plot([4, 4], [y+0.4, y-0.8], 'k-', linewidth=1)
        
        # Bottleneck
        bottleneck = FancyBboxPatch((1.5, 1), (6.5, 0.5), 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['conv'], 
                                edgecolor='black', linewidth=2)
        ax.add_patch(bottleneck)
        ax.text(7.25, 1, 'Bottleneck\n1024', 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Decoder path
        decoder_levels = [2, 4, 6, 8]
        for i, y in enumerate(decoder_levels):
            # Upsample
            up = FancyBboxPatch((0.8, 0.4), (6.5, y), 
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['upsample'], 
                              edgecolor='black', linewidth=1)
            ax.add_patch(up)
            ax.text(6.9, y+0.2, 'UpConv\n2x2x2', 
                    ha='center', va='center', fontsize=7)
            
            # Conv blocks (no attention in hybrid)
            conv = FancyBboxPatch((1, 0.8), (8, y), 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['conv'], 
                                edgecolor='black', linewidth=1)
            ax.add_patch(conv)
            ax.text(8.5, y+0.4, f'Conv\n{256//2**i}', 
                    ha='center', va='center', fontsize=8)
            
            # Skip connections
            if i < 3:
                skip_y = encoder_levels[2-i]
                ax.plot([1.5, 8], [skip_y+0.4, y+0.4], 'r--', linewidth=2, alpha=0.7)
                ax.plot([2.5, 8], [skip_y+0.4, y+0.4], 'r--', linewidth=2, alpha=0.7)
        
        # Output
        output = FancyBboxPatch((1, 0.8), (6.5, 9), 
                              boxstyle="round,pad=0.1", 
                              facecolor='yellow', 
                              edgecolor='black', linewidth=2)
        ax.add_patch(output)
        ax.text(7, 9.4, 'Output\n3 classes', 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Final connections
        ax.plot([7.25, 8.5], [1.5, 9], 'k-', linewidth=2)
        ax.plot([7.25, 8.5], [7.5, 9], 'k-', linewidth=2)
    
    plt.tight_layout()
    plt.savefig('results/architecture_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Architecture diagram saved to results/architecture_diagram.png")

def create_method_flowchart():
    """Create flowchart of the proposed method"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_title('Proposed Hybrid Attention Method', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Input
    input_box = FancyBboxPatch((2, 1), (1, 8), 
                             boxstyle="round,pad=0.1", 
                             facecolor='lightblue', 
                             edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2, 8.5, '3D MRI Input\n(4 modalities)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Encoder with SE
    encoder_box = FancyBboxPatch((2.5, 1.5), (4, 6), 
                               boxstyle="round,pad=0.1", 
                               facecolor='orange', 
                               edgecolor='black', linewidth=2)
    ax.add_patch(encoder_box)
    ax.text(5.25, 6.75, 'Encoder\n+ SE Attention', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Bottleneck with CBAM
    bottleneck_box = FancyBboxPatch((2, 1.5), (7, 4), 
                                 boxstyle="round,pad=0.1", 
                                 facecolor='lightgreen', 
                                 edgecolor='black', linewidth=2)
    ax.add_patch(bottleneck_box)
    ax.text(8, 4.75, 'Bottleneck\n+ CBAM Attention', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Decoder (no attention)
    decoder_box = FancyBboxPatch((2.5, 1.5), (4, 2), 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightblue', 
                              edgecolor='black', linewidth=2)
    ax.add_patch(decoder_box)
    ax.text(5.25, 2.75, 'Decoder\n(No Attention)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Output
    output_box = FancyBboxPatch((2, 1), (7, 0), 
                             boxstyle="round,pad=0.1", 
                             facecolor='yellow', 
                             edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(8, 0.5, 'Segmentation\nOutput', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    ax.annotate('', xy=(4, 6), xytext=(2, 7.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(7, 4), xytext=(6.5, 5.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(5.25, 2), xytext=(8, 3.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(7, 0.5), xytext=(5.25, 1.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Key insights
    ax.text(1, 9, 'Key Insights:', fontsize=12, fontweight='bold')
    ax.text(1, 8.5, '• SE in encoder: Channel-wise feature selection', fontsize=9)
    ax.text(1, 8, '• CBAM at bottleneck: Spatial reasoning', fontsize=9)
    ax.text(1, 7.5, '• No decoder attention: Preserve reconstruction', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/method_flowchart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Method flowchart saved to results/method_flowchart.png")

if __name__ == "__main__":
    from pathlib import Path
    Path("results").mkdir(exist_ok=True)
    
    create_architecture_diagram()
    create_method_flowchart()
    print("🎯 Architecture diagrams generated for paper!")
