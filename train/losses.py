import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy loss.
    
    Args:
        pos_weight (float): Weight for positive class (y_true == 1).
        null_weight (float): Weight for negative class (y_true == 0).
    """
    def __init__(self, pos_weight, null_weight):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.null_weight = null_weight

    def forward(self, y_pred, y_true):
        """
        Computes the weighted binary cross-entropy loss.

        Args:
            y_pred (Tensor): Predicted logits (before applying sigmoid).
            y_true (Tensor): Ground truth binary labels (0 or 1).

        Returns:
            Tensor: Computed loss value.
        """
        # Assign weights based on the ground truth labels
        weight = torch.where(y_true == 0, self.null_weight, self.pos_weight)
        # Compute the binary cross-entropy loss with the specified weights (includes sigmoid)
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true, weight=weight)
        return loss