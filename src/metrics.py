import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff

def compute_hd95_per_class(logits, target, eps=1e-6):
    """
    Compute 95th percentile Hausdorff Distance for each class
    """
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)  # [B, D, H, W]
        num_classes = logits.shape[1]
        hd95_scores = []
        
        for c in range(1, num_classes):  # skip background
            pred_c = (preds == c).float()
            targ_c = (target == c).float()
            
            # Convert to CPU numpy
            pred_np = pred_c.cpu().numpy()
            targ_np = targ_c.cpu().numpy()
            
            # Compute HD95 for this batch element
            batch_hd95 = []
            for b in range(pred_np.shape[0]):
                if pred_np[b].sum() > 0 and targ_np[b].sum() > 0:
                    # Get surface points (non-zero voxels)
                    pred_points = np.argwhere(pred_np[b] > 0)
                    targ_points = np.argwhere(targ_np[b] > 0)
                    
                    # Compute directed Hausdorff distances
                    dist1 = directed_hausdorff(pred_points, targ_points)[0]
                    dist2 = directed_hausdorff(targ_points, pred_points)[0]
                    hd = max(dist1, dist2)
                    hd95 = np.percentile([dist1, dist2], 95)
                    batch_hd95.append(hd95)
                else:
                    # Handle empty masks
                    batch_hd95.append(0.0)
            
            hd95_scores.append(np.mean(batch_hd95))
    
    return hd95_scores

def compute_dice_per_class(logits, target, eps=1e-6):
    """
    Compute Dice coefficient for each class
    """
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

def compute_brats_regions_metrics(logits, target, eps=1e-6):
    """
    Compute Dice and HD95 for BraTS regions: WT, TC, ET
    
    BraTS regions:
    - WT (Whole Tumor): labels 1 + 2 + 4 (tumor-core + edema + enhancing)
    - TC (Tumor Core): labels 1 + 4 (tumor-core + enhancing)  
    - ET (Enhancing Tumor): label 4 only
    
    Our model outputs: 0 (background), 1 (tumor-core), 2 (edema)
    Need to map back to original BraTS labels for region calculation
    """
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)  # [B, D, H, W] - values 0,1,2
        
        # Convert model predictions back to BraTS labels
        # Model: 0=bg, 1=TC, 2=ED
        # BraTS: 0=bg, 1=TC, 2=ED, 4=ET
        pred_brats = preds.clone()
        pred_brats[pred_brats == 2] = 2  # edema stays 2
        # Note: model doesn't predict ET (4) directly, ET is part of TC in our 3-class setup
        
        # Convert target back to BraTS labels  
        target_brats = target.clone()
        
        # Compute regions
        results = {}
        
        # Whole Tumor (WT): labels 1 + 2 + 4 (in our case: 1 + 2)
        pred_wt = (pred_brats == 1) | (pred_brats == 2)
        targ_wt = (target_brats == 1) | (target_brats == 2)
        results['wt_dice'] = compute_dice_binary(pred_wt, targ_wt, eps)
        results['wt_hd95'] = compute_hd95_binary(pred_wt, targ_wt)
        
        # Tumor Core (TC): labels 1 + 4 (in our case: just 1)
        pred_tc = (pred_brats == 1)
        targ_tc = (target_brats == 1)
        results['tc_dice'] = compute_dice_binary(pred_tc, targ_tc, eps)
        results['tc_hd95'] = compute_hd95_binary(pred_tc, targ_tc)
        
        # Enhancing Tumor (ET): label 4 only
        # Since we don't predict ET separately, ET will be 0
        pred_et = torch.zeros_like(pred_brats, dtype=torch.bool)
        targ_et = (target_brats == 4)  # Original ET labels
        results['et_dice'] = compute_dice_binary(pred_et, targ_et, eps)
        results['et_hd95'] = compute_hd95_binary(pred_et, targ_et)
        
        return results

def compute_dice_binary(pred_mask, targ_mask, eps=1e-6):
    """Compute Dice for binary masks"""
    pred_float = pred_mask.float()
    targ_float = targ_mask.float()
    inter = (pred_float * targ_float).sum()
    union = pred_float.sum() + targ_float.sum()
    if union == 0:
        return 0.0
    dice = (2*inter + eps) / (union + eps)
    return dice.item()

def compute_hd95_binary(pred_mask, targ_mask):
    """Compute HD95 for binary masks"""
    pred_np = pred_mask.cpu().numpy()
    targ_np = targ_mask.cpu().numpy()
    
    batch_hd95 = []
    for b in range(pred_np.shape[0]):
        if pred_np[b].sum() > 0 and targ_np[b].sum() > 0:
            pred_points = np.argwhere(pred_np[b] > 0)
            targ_points = np.argwhere(targ_np[b] > 0)
            
            dist1 = directed_hausdorff(pred_points, targ_points)[0]
            dist2 = directed_hausdorff(targ_points, pred_points)[0]
            hd95 = np.percentile([dist1, dist2], 95)
            batch_hd95.append(hd95)
        else:
            batch_hd95.append(0.0)
    
    return np.mean(batch_hd95)
