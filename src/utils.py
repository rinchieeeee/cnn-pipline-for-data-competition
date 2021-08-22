import torch
import numpy as np

def make_mixup(image, labels, config, cuda = True):
    
    # defalut is 1.0
    alpha = config["alpha"]
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = labels.size(0)
    if cuda:
        idx = torch.randperm(batch_size).cuda()
    else:
        idx = torch.randperm(batch_size)

    mixed_image = lam * image + (1 - lam) * image[idx, :]
    y_1, y_2 = labels, labels[idx]

    return mixed_image, y_1, y_2, lam


def mixup_critetion(loss_func, y_pred, y_1, y_2, lam):
    return lam * loss_func(y_pred, y_1) + (1 - lam) * loss_func(y_pred, y_2)