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
from dataset_torchio import make_loaders  # expects this module in src/
from metrics import compute_dice_per_class, compute_hd95_per_class

# TensorBoard: optional import
writer = None
def init_tensorboard(log_dir, enable):
    global writer
    if enable:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=log_dir)
        except Exception as e:
            print(f"TensorBoard disabled due to import error: {e}")
            writer = None
    else:
        writer = None

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def preprocess_mask(mask):
    """
    Convert BraTS segmentation labels to 3-class format:
    - 0 -> 0 (background)
    - 1 -> 1 (tumor-core)
    - 2 -> 2 (edema)  
    - 4 -> 1 (tumor-core)
    mask: tensor [B, D, H, W]
    """
    mask_np = mask.clone()
    # Convert BraTS labels to 3-class format
    mask_np[mask == 4] = 1  # enhancing tumor -> tumor core
    # Values 0, 1, 2 are already correct
    mask_np[mask_np > 2] = 2  # safety: any remaining values > 2 -> edema
    return mask_np.long()

def compute_dice_per_class(logits, target, eps=1e-6):
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)  # [B, D, H, W]
        num_classes = logits.shape[1]
        dices = []
        for c in range(1, num_classes):  # skip background for reporting (optional)
            pred_c = (preds == c).float()
            targ_c = (target == c).float()
            inter = (pred_c * targ_c).sum()
            union = pred_c.sum() + targ_c.sum()
            dice = (2*inter + eps) / (union + eps)
            dices.append(dice.item())
    return dices

def save_checkpoint(state, is_best, out_dir, filename='checkpoint.pth'):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    torch.save(state, os.path.join(out_dir, filename))
    if is_best:
        torch.save(state, os.path.join(out_dir, 'best.pth'))

def train_one_epoch(model, loader, optimizer, scaler, device, loss_fn, epoch, writer=None):
    model.train()
    pbar = tqdm(loader, desc=f"Train E{epoch}")
    running_loss = 0.0
    for i, (images, masks) in enumerate(pbar):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.long)
        masks = remap_mask(masks)

        optimizer.zero_grad()
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
        if writer is not None:
            writer.add_scalar('train/loss_batch', loss.item(), epoch * len(loader) + i)
        pbar.set_postfix(loss=running_loss / (i+1))
    return running_loss / len(loader)

def validate(model, loader, device, loss_fn, epoch, writer=None):
    model.eval()
    running_loss = 0.0
    dices_accum = []
    hd95_accum = []
    pbar = tqdm(loader, desc=f"Val E{epoch}")
    with torch.no_grad():
        for i, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            logits = model(images)
            loss = loss_fn(logits, masks)
            running_loss += loss.item()
            
            dices = compute_dice_per_class(logits, masks)
            hd95_scores = compute_hd95_per_class(logits, masks)
            
            dices_accum.append(dices)
            hd95_accum.append(hd95_scores)
            pbar.set_postfix(loss=running_loss / (i+1))

    avg_loss = running_loss / len(loader)
    # compute mean dice across dataset for each class
    dices_arr = np.array(dices_accum)
    hd95_arr = np.array(hd95_accum)
    
    if dices_arr.size == 0:
        mean_dices = [0.0, 0.0]
        mean_hd95 = [0.0, 0.0]
    else:
        mean_dices = dices_arr.mean(axis=0).tolist()
        mean_hd95 = hd95_arr.mean(axis=0).tolist()
        
    if writer is not None:
        writer.add_scalar('val/loss', avg_loss, epoch)
        writer.add_scalar('val/dice_class1', mean_dices[0], epoch)
        writer.add_scalar('val/dice_class2', mean_dices[1], epoch)
        writer.add_scalar('val/hd95_class1', mean_hd95[0], epoch)
        writer.add_scalar('val/hd95_class2', mean_hd95[1], epoch)
    return avg_loss, mean_dices, mean_hd95

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--preproc', type=str, required=True, help='preprocessed .npz folder')
    p.add_argument('--outdir', type=str, default='outputs', help='output folder')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--max_batches', type=int, default=None, help='for smoke test: limit number of cases used (int)')
    p.add_argument('--amp', action='store_true', help='use mixed precision')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--smoke', action='store_true', help='quick smoke test: 1 epoch, small subset')
    p.add_argument('--no_tensorboard', action='store_true', help='disable TensorBoard logging')
    p.add_argument('--scheduler', type=str, default='cosine', choices=['none', 'cosine', 'step'], help='LR scheduler')
    p.add_argument('--attention', type=str, default='none', choices=['none', 'se', 'cbam', 'hybrid'], help='attention mechanism type')
    return p.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    init_tensorboard(os.path.join(args.outdir, 'tb'), not args.no_tensorboard)

    # Make data loaders (uses dataset_torchio.make_loaders)
    train_loader, val_loader = make_loaders(
        args.preproc,
        batch_size=args.batch,
        num_workers=args.workers,
        shuffle_train=True,
        augment=True
    )

    # If max_batches (smoke) provided, wrap samplers or reduce dataset length by monkey patching
    if args.max_batches:
        # naive: create small index subsets (works because make_loaders uses Subset samplers)
        print("WARNING: --max_batches option: not guaranteed to limit dataset size if make_loaders changed.")
        # Could implement custom sampler; for now user may use preprocess --max to create small set.

    model = UNet3D(in_channels=4, base_filters=16, num_classes=3, attention_type=args.attention)  # smaller base_filters for speed
    model = model.to(device)
    
    print(f"Training model with attention: {args.attention}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Class-balanced weights: inverse frequency for BraTS (background=0, tumor-core=1, edema=2)
    class_weights = torch.tensor([0.1, 1.0, 1.0]).to(device)
    loss_fn = DiceCELoss(weight_ce=1.0, weight_dice=1.0, class_weights=class_weights)

    # Scheduler
    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs//3, gamma=0.1)

    scaler = torch.cuda.amp.GradScaler() if (args.amp and torch.cuda.is_available()) else None

    best_val = 1e9
    start_time = time.time()

    epochs = 1 if args.smoke else args.epochs
    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, loss_fn, epoch, writer)
        val_loss, val_dices, val_hd95 = validate(model, val_loader, device, loss_fn, epoch, writer)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dices={val_dices} val_hd95={val_hd95}")

        if scheduler is not None:
            scheduler.step()
            if writer is not None:
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)

        is_best = val_loss < best_val
        best_val = min(val_loss, best_val)
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }, is_best, args.outdir)

    if writer is not None:
        writer.close()
    print("Training finished in", time.time() - start_time)

if __name__ == '__main__':
    main()
