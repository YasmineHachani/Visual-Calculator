import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Vgg11(nn.Module):
    def __init__(self, pretrained_backbone=False, **kwargs):
        super().__init__()
        print(f'Using pretraining : {pretrained_backbone}')
        self.m = models.vgg11(pretrained=pretrained_backbone)
        self.out_features = self.m.fc.in_features

    def forward(self, x):
        """
        Prediction

        Parameters
        ----------
        x : tensor (NBImages, C, W , H)

        Returns
        -------
        prediction : tensor (NBImages, NBClass)

        """
        return self.m(x)
