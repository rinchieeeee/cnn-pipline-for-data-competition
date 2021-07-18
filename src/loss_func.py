import torch
import torch.nn as nn

def get_lossfunc(loss_name):

    if loss_name == "BCEWithLogitLoss":
        return nn.BCEWithLogitLoss()