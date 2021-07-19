import torch
import torch.nn as nn

def get_lossfunc(loss_name):

    if loss_name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()