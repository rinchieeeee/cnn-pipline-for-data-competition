import torch
import torch.nn as nn
from arcface import ArcFaceLoss


class FocalLoss(nn.Module):
    """
    Reference:
        https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution/blob/master/src/myloss/loss.py
    """

    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        loss_bce = self.bce_loss(inputs, targets)
        pt = torch.exp(-loss_bce)  # get probabilities model predicts
        loss_f = self.alpha * (torch.tensor(1.0) - pt) ** self.gamma * loss_bce
        return loss_f.mean()


def get_lossfunc(loss_name):

    if loss_name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()

    elif loss_name == "FocalLoss":
        return FocalLoss()

    elif loss_name == "ArcFaceLoss":
        return ArcFaceLoss()