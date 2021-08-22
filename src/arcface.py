import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class ArcFaceLoss(nn.Module):

    def __init__(self, scale = 26.0, mergin = 0.5):
        """
        scale: parameter "s" in Arcface paper. Feature scale parameter
        mergin: parameter "m" in ArcFace paper. 
        """
        super(ArcFaceLoss, self).__init__()
        self.scale = scale
        self.mergin = mergin
        self.arccos = torch.arccos
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        """
        logits: feature through by ArcFaceLayer Module. e.g. logits' value is cosine
        """
        one_hot = torch.zeros(logits.size(), device = "cuda")
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1) # make one-hot vector
        theta = torch.arccos(logits) # 角度を取得
        theta = theta + self.mergin * one_hot # 正解ラベルに merginが足された角度
        output = torch.cos(theta)
        output *= self.scale
        return self.cross_entropy(output, labels.long())

class ArcFaceLayer(nn.Module):

    def __init__(self, in_features, class_num):
        super(ArcFaceLayer, self).__init__()
        # nn.Parameter()で囲むことで, パラメータ更新対象のパラメータとなる
        self.weights = nn.Parameter(torch.FloatTensor(class_num, in_features))
        self.weight_init()

    def weight_init(self):
        std = math.sqrt(6.0/self.weights.size(1)) # Heの初期化. weights.size(1) = in_features
        self.weights.data.uniform_(-std, std)

    def forward(self, x):
        """
        x: x is features after through encoder -> Global Pooling and flatten
        """
        # inner product
        cosine = F.linear(F.normalize(x, p = 2.0), F.normalize(self.weights.cuda(), p = 2.0), bias = None) 
        return cosine
