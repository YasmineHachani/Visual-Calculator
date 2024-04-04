import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleConv(nn.Module):
    def __init__(self, NBClass):
        super().__init__()

         # First convolutional layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.LayerNorm((16, 224, 224))  # Layer normalization for the first layer
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.LayerNorm((32, 112, 112))  # Layer normalization for the second layer
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.norm3 = nn.LayerNorm((256, 56, 56))
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.norm4 = nn.LayerNorm((512, 28, 28))
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(self.out_features, NBClass)
        

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
        
        output = self.pool1(self.relu1(self.norm1(self.conv1(x))))
        output = self.pool2(self.relu2(self.norm2(self.conv2(output))))
        output = self.pool3(self.relu3(self.norm3(self.conv3(output))))
        output = self.pool4(self.relu4(self.norm4(self.conv4(output))))
        output = self.fc(output)
        
        return output


