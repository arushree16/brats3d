#!/usr/bin/env python3
"""
Colab-optimized training script for faster baseline training
"""

import os
import sys
import time
import random
from pathlib import Path
import argparse
import numpy as np

import torch
from torch import nn, optim
from tqdm import tqdm

# Ensure src is importable
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from unet3d import UNet3D
from losses import DiceCELoss
from dataset_torchio import make_loaders
from metrics import compute_dice_per_class, compute_hd95_per_class, compute_brats_regions_metrics

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def preprocess_mask(mask):
    mask_np = mask.clone()
    mask_np[mask == 4] = 1
    mask_np[mask_np > 2] = 2
    return mask_np.long()

def save_checkpoint(state, is_best, out_dir, filename='checkpoint.pth'):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    torch.save(state, os.path.join(out_dir, filename))
    if is_best:
        torch.save(state, os.path.join(out_dir, 'best.pth'))

def train_one_epoch(model, loader, optimizer, scaler, device, loss_fn, epoch):
    model.train()
    pbar = tqdm(loader, desc=f"Train E{epoch}")
    running_loss = 0.0
    
    for i, (images, masks) in enumerate(pbar):
        images = images.to(device, dtype=torch.float32, non_blocking=True)
        masks = masks.to(device, dtype=torch.long, non_blocking=True)
        masks = preprocess_mask(masks)

        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(images)
            loss = loss_fn(logits, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=running_loss / (i+1))
        
        # Clear cache periodically for Colab
        if i % 10 == 0:
            torch.cuda.empty_cache()
            
    return running_loss / len(loader)

def validate(model, loader, device, loss_fn, epoch):
    model.eval()
    running_loss = 0.0
    dices_accum = []
    brats_accum = []
    pbar = tqdm(loader, desc=f"Val E{epoch}")
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(pbar):
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            masks = masks.to(device, dtype=torch.long, non_blocking=True)
            masks = preprocess_mask(masks)
            
            logits = model(images)
            loss = loss_fn(logits, masks)
            running_loss += loss.item()
            
            dices = compute_dice_per_class(logits, masks)
            brats_metrics = compute_brats_regions_metrics(logits, masks)
            
            dices_accum.append(dices)
            brats_accum.append(brats_metrics)
            pbar.set_postfix(loss=running_loss / (i+1))

    avg_loss = running_loss / len(loader)
    dices_arr = np.array(dices_accum)
    mean_dices = dices_arr.mean(axis=0).tolist() if dices_arr.size > 0 else [0.0, 0.0]
    
    brats_wt_dice = np.mean([b['wt_dice'] for b in brats_accum])
    brats_tc_dice = np.mean([b['tc_dice'] for b in brats_accum])
    
    brats_metrics = {
        'wt_dice': brats_wt_dice,
        'tc_dice': brats_tc_dice
    }
    
    return avg_loss, mean_dices, brats_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preproc', type=str, required=True)
    parser.add_argument('--outdir', type=str, default='outputs')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-4)  # Higher LR for faster convergence
    parser.add_argument('--attention', type=str, default='none', choices=['none', 'se', 'cbam', 'hybrid'])
    parser.add_argument('--amp', action='store_true', default=True)  # Enable AMP by default
    args = parser.parse_args()
    
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    
    print(f"Training on {device}")
    print(f"Batch size: {args.batch}")
    print(f"Mixed precision: {args.amp}")

    # Optimized data loading
    train_loader, val_loader = make_loaders(
        args.preproc,
        batch_size=args.batch,
        num_workers=0,  # Colab optimization
        shuffle_train=True,
        augment=False  # Disable for baseline speed test
    )

    # Model with optimized parameters
    model = UNet3D(in_channels=4, base_filters=16, num_classes=3, attention_type=args.attention)
    model = model.to(device)
    
    # Enable optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    class_weights = torch.tensor([0.1, 1.0, 1.0]).to(device)
    loss_fn = DiceCELoss(weight_ce=1.0, weight_dice=1.0, class_weights=class_weights)
    
    # Cosine scheduler for faster convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    best_val = 1e9
    start_time = time.time()

    for epoch in range(1, args.epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, loss_fn, epoch)
        val_loss, val_dices, val_brats = validate(model, val_loader, device, loss_fn, epoch)
        
        scheduler.step()
        
        print(f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} lr={scheduler.get_last_lr()[0]:.6f}")
        print(f"  Dice: TC={val_dices[0]:.3f}, ED={val_dices[1]:.3f}")
        print(f"  BraTS: WT={val_brats['wt_dice']:.3f}, TC={val_brats['tc_dice']:.3f}")

        is_best = val_loss < best_val
        best_val = min(val_loss, best_val)
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_brats': val_brats
        }, is_best, args.outdir)

    print(f"Training finished in {time.time() - start_time:.1f}s")

if __name__ == '__main__':
    main()
