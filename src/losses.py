import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6, softmax=True):
        super().__init__()
        self.eps = eps
        self.softmax = softmax

    def forward(self, logits, target):
        """
        logits: [B, C, D, H, W] (raw logits)
        target: [B, D, H, W] (0..C-1)
        """
        if self.softmax:
            probs = F.softmax(logits, dim=1)
        else:
            probs = logits

        B, C = probs.shape[0], probs.shape[1]
        probs = probs.view(B, C, -1)
        # one-hot encode target
        with torch.no_grad():
            target_flat = target.view(B, -1).long()  # [B, N]
            target_onehot = F.one_hot(target_flat, num_classes=C).float()  # [B, N, C]
            target_onehot = target_onehot.permute(0, 2, 1)  # [B, C, N]
        intersection = (probs * target_onehot).sum(-1)
        cardinality = probs.sum(-1) + target_onehot.sum(-1)
        dices = (2. * intersection + self.eps) / (cardinality + self.eps)
        # average over classes and batch
        loss = 1.0 - dices.mean()
        return loss

class DiceCELoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0, ignore_index=-100, class_weights=None):
        super().__init__()
        if class_weights is not None:
            self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss()
        self.w_ce = weight_ce
        self.w_dice = weight_dice

    def forward(self, logits, target):
        l_ce = self.ce(logits, target.long())
        l_dice = self.dice(logits, target)
        return self.w_ce * l_ce + self.w_dice * l_dice
