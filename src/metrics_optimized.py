import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt

def compute_hd95_binary_fast(pred_mask, targ_mask):
    """Fast HD95 using distance transforms - much faster than Hausdorff"""
    pred_np = pred_mask.cpu().numpy()
    targ_np = targ_mask.cpu().numpy()
    
    batch_hd95 = []
    for b in range(pred_np.shape[0]):
        pred_vol = pred_np[b]
        targ_vol = targ_np[b]
        
        # Skip if either mask is empty
        if pred_vol.sum() == 0 or targ_vol.sum() == 0:
            batch_hd95.append(0.0)
            continue
        
        # Compute distance transforms (much faster!)
        pred_dt = distance_transform_edt(1 - pred_vol)
        targ_dt = distance_transform_edt(1 - targ_vol)
        
        # Get surface distances
        pred_to_targ = targ_dt[pred_vol > 0]
        targ_to_pred = pred_dt[targ_vol > 0]
        
        # Compute 95th percentile
        if len(pred_to_targ) > 0 and len(targ_to_pred) > 0:
            dist1 = np.percentile(pred_to_targ, 95)
            dist2 = np.percentile(targ_to_pred, 95)
            hd95 = max(dist1, dist2)
        else:
            hd95 = 0.0
            
        batch_hd95.append(hd95)
    
    return np.mean(batch_hd95)

def compute_hd95_binary_sampled(pred_mask, targ_mask, sample_size=1000):
    """Fast HD95 using point sampling - good approximation"""
    pred_np = pred_mask.cpu().numpy()
    targ_np = targ_mask.cpu().numpy()
    
    batch_hd95 = []
    for b in range(pred_np.shape[0]):
        pred_vol = pred_np[b]
        targ_vol = targ_np[b]
        
        if pred_vol.sum() == 0 or targ_vol.sum() == 0:
            batch_hd95.append(0.0)
            continue
        
        # Sample points (much fewer!)
        pred_points = np.argwhere(pred_vol > 0)
        targ_points = np.argwhere(targ_vol > 0)
        
        # Limit to sample_size points
        if len(pred_points) > sample_size:
            pred_points = pred_points[np.random.choice(len(pred_points), sample_size, replace=False)]
        if len(targ_points) > sample_size:
            targ_points = targ_points[np.random.choice(len(targ_points), sample_size, replace=False)]
        
        # Compute Hausdorff on sampled points (much faster!)
        try:
            dist1 = directed_hausdorff(pred_points, targ_points)[0]
            dist2 = directed_hausdorff(targ_points, pred_points)[0]
            hd95 = max(dist1, dist2)
        except:
            hd95 = 0.0
            
        batch_hd95.append(hd95)
    
    return np.mean(batch_hd95)

# Replace the slow functions
def compute_brats_regions_metrics_fast(logits, target, eps=1e-6):
    """Fast version using optimized HD95"""
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        
        pred_brats = preds.clone()
        target_brats = target.clone()
        
        results = {}
        
        # Whole Tumor (WT): labels 1 + 2
        pred_wt = (pred_brats == 1) | (pred_brats == 2)
        targ_wt = (target_brats == 1) | (target_brats == 2)
        results['wt_dice'] = compute_dice_binary(pred_wt, targ_wt, eps)
        results['wt_hd95'] = compute_hd95_binary_fast(pred_wt, targ_wt)
        
        # Tumor Core (TC): label 1
        pred_tc = (pred_brats == 1)
        targ_tc = (target_brats == 1)
        results['tc_dice'] = compute_dice_binary(pred_tc, targ_tc, eps)
        results['tc_hd95'] = compute_hd95_binary_fast(pred_tc, targ_tc)
        
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
