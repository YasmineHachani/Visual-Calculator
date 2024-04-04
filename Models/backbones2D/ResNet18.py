import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet18(nn.Module):
    def __init__(self, NBClass, pretrained_backbone=False, **kwargs):
        super().__init__()
        print(f'Using pretraining : {pretrained_backbone}')
        m = models.resnet18(pretrained=pretrained_backbone)
        self.out_features = m.fc.in_features
        self.feature_extractor = nn.Sequential(*list(m.children())[:-1])
        self.head = nn.Linear(self.out_features, NBClass)

    def forward(self, x):
        """
        Prediction

        Parameters
        ----------
        batch : Dict containing 'image' (NBImages, C, W , H)

        Returns
        -------
        prediction : tensor (NBImages, NBClass)

        """
        features = self.feature_extractor(x)
        features = features.reshape(features.size(0), -1)
        return self.head(features)
