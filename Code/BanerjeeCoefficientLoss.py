import torch
import torch.nn as nn
import torch.nn.functional as F

class BanerjeeCoefficientLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=2.0, smooth=1e-6):
        """
        Initializes the Banerjee Coefficient Loss.

        Parameters:
        - alpha (float): Weight for |P - G|
        - beta (float): Weight for |G - P|
        - gamma (float): Exponent for the third term
        - smooth (float): Smoothing factor to prevent division by zero
        """
        super(BanerjeeCoefficientLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        Computes the Banerjee Coefficient Loss.

        Parameters:
        - preds: Predicted tensor (logits or probabilities), shape (B, 1, H, W)
        - targets: Ground truth tensor, shape (B, 1, H, W)

        Returns:
        - loss: scalar tensor of loss value
        """
        preds = torch.sigmoid(preds)  # Ensure predictions are between 0 and 1
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (preds * targets).sum(dim=1)
        P_minus_G = ((preds - targets).clamp(min=0)).sum(dim=1)
        G_minus_P = ((targets - preds).clamp(min=0)).sum(dim=1)
        P_plus_G = preds.sum(dim=1) + targets.sum(dim=1)

        # Term 1: Modified Tversky coefficient
        tversky_base = intersection / (intersection + self.alpha * P_minus_G + self.beta * G_minus_P + self.smooth)
        term1 = 0.23 * (1 - tversky_base)

        # Term 2: Dice coefficient
        dice = (2 * intersection + self.smooth) / (P_plus_G + self.smooth)
        term2 = 0.65 * (1 - dice)

        # Term 3: Exponential Tversky-based loss
        term3 = 0.12 * torch.pow((1 - tversky_base), self.gamma)

        total_loss = term1 + term2 + term3
        return total_loss.mean()
