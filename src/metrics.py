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
