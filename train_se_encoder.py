#!/usr/bin/env python3
"""
Training script for SE-Encoder Only Model
SE blocks only in encoder, no attention in decoder or bottleneck
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import json
import os
from pathlib import Path
import time
from tqdm import tqdm
import sys

# Ensure src is importable
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from unet3d import UNet3D
from dataset_torchio import make_loaders
from metrics_optimized import compute_brats_regions_metrics_fast

def train_se_encoder_only():
    """Train SE-Encoder Only model"""
    
    print("🧠 Training SE-Encoder Only Model")
    print("=" * 50)
    
    # Configuration
    config = {
        'model_type': 'se_encoder_only',
        'in_channels': 4,
        'base_filters': 32,
        'num_classes': 3,
        'batch_size': 2,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'data_dir': 'data/processed/brats128',
        'output_dir': 'outputs/se_encoder_only'
    }
    
    print(f"📊 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Load data
    print("\n📁 Loading data...")
    train_loader, val_loader = make_loaders(
        config['data_dir'], 
        batch_size=config['batch_size'], 
        augment=True
    )
    print(f"✅ Training samples: {len(train_loader.dataset)}")
    print(f"✅ Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("\n🏗️ Creating SE-Encoder Only model...")
    model = UNet3D(
        in_channels=config['in_channels'],
        base_filters=config['base_filters'],
        num_classes=config['num_classes'],
        attention_type='se_encoder_only'  # Custom attention type
    ).to(config['device'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 Total parameters: {total_params:,}")
    print(f"📈 Trainable parameters: {trainable_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scaler = GradScaler('cuda')
    
    # Training log
    training_log = {
        'train_loss': [],
        'val_loss': [],
        'wt_dice': [],
        'tc_dice': [],
        'wt_hd95': [],
        'tc_hd95': [],
        'config': config,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params
    }
    
    print("\n🚀 Starting training...")
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
        for batch_idx, (images, masks) in enumerate(train_pbar):
            images, masks = images.to(config['device']), masks.to(config['device'])
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            train_samples += images.size(0)
            
            # Update progress bar
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / train_samples
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_samples = 0
        val_metrics = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Val]")
            for images, masks in val_pbar:
                images, masks = images.to(config['device']), masks.to(config['device'])
                
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                
                val_loss += loss.item() * images.size(0)
                val_samples += images.size(0)
                
                # Calculate metrics
                pred = torch.argmax(outputs, dim=1)
                brats_metrics = compute_brats_regions_metrics_fast(pred, masks)
                val_metrics.append(brats_metrics)
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / val_samples
        
        # Aggregate validation metrics
        avg_wt_dice = np.mean([m['wt_dice'] for m in val_metrics])
        avg_tc_dice = np.mean([m['tc_dice'] for m in val_metrics])
        avg_wt_hd95 = np.mean([m['wt_hd95'] for m in val_metrics])
        avg_tc_hd95 = np.mean([m['tc_hd95'] for m in val_metrics])
        
        # Update training log
        training_log['train_loss'].append(avg_train_loss)
        training_log['val_loss'].append(avg_val_loss)
        training_log['wt_dice'].append(avg_wt_dice)
        training_log['tc_dice'].append(avg_tc_dice)
        training_log['wt_hd95'].append(avg_wt_hd95)
        training_log['tc_hd95'].append(avg_tc_hd95)
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  WT Dice: {avg_wt_dice:.4f}, TC Dice: {avg_tc_dice:.4f}")
        print(f"  WT HD95: {avg_wt_hd95:.1f}, TC HD95: {avg_tc_hd95:.1f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': config
            }, os.path.join(config['output_dir'], 'best.pth'))
            print(f"  💾 New best model saved!")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': config
            }, os.path.join(config['output_dir'], f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Calculate final metrics
    total_time = time.time() - start_time
    final_metrics = {
        'final_train_loss': training_log['train_loss'][-1],
        'final_val_loss': training_log['val_loss'][-1],
        'final_wt_dice': training_log['wt_dice'][-1],
        'final_tc_dice': training_log['tc_dice'][-1],
        'final_wt_hd95': training_log['wt_hd95'][-1],
        'final_tc_hd95': training_log['tc_hd95'][-1],
        'total_training_time': total_time,
        'best_val_loss': best_val_loss
    }
    
    training_log.update(final_metrics)
    
    # Save training log
    with open(os.path.join(config['output_dir'], 'training_log.json'), 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\n🎉 Training completed!")
    print(f"⏱️ Total time: {total_time/3600:.2f} hours")
    print(f"📊 Final WT Dice: {final_metrics['final_wt_dice']:.4f}")
    print(f"📊 Final TC Dice: {final_metrics['final_tc_dice']:.4f}")
    print(f"💾 Results saved to: {config['output_dir']}")
    
    return training_log

if __name__ == "__main__":
    train_se_encoder_only()
