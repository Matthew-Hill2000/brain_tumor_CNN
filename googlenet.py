import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import random
from PIL import Image
import os

class Inception(nn.Module):
    """
    Inception block module, inspired by the Inception architecture.

    This module is composed of four parallel paths:
    1. A 1x1 convolution.
    2. A 1x1 convolution followed by a 3x3 convolution.
    3. A 1x1 convolution followed by a 5x5 convolution.
    4. A 3x3 max pooling followed by a 1x1 convolution.

    The outputs of these paths are concatenated along the channel dimension.

    Args:
        output_1 (int): Number of output channels for the first path (1x1 convolution).
        output_2 (tuple): Number of output channels for the second path:
                          - output_2[0]: Number of output channels for the 1x1 convolution.
                          - output_2[1]: Number of output channels for the 3x3 convolution.
        output_3 (tuple): Number of output channels for the third path:
                          - output_3[0]: Number of output channels for the 1x1 convolution.
                          - output_3[1]: Number of output channels for the 5x5 convolution.
        output_4 (int): Number of output channels for the fourth path (3x3 max pooling followed by 1x1 convolution).

    Methods:
        forward(x):
            Defines the computation performed at every call. The input is processed through four branches, 
            and their outputs are concatenated along the channel dimension.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            Returns:
                torch.Tensor: Output tensor after processing through the Inception block.
    """
    def __init__(self, output_1, output_2, output_3, output_4):
        super(Inception, self).__init__()

        self.ReLU = nn.ReLU()

        self.conv_one_branch_one = nn.LazyConv2d(out_channels=output_1, kernel_size=1)

        self.conv_one_branch_two = nn.LazyConv2d(out_channels=output_2[0], kernel_size=1)
        self.conv_two_branch_two = nn.LazyConv2d(out_channels=output_2[1], kernel_size=3, padding=1)

        self.conv_one_branch_three = nn.LazyConv2d(out_channels=output_3[0], kernel_size=1)
        self.conv_two_branch_three = nn.LazyConv2d(out_channels=output_3[1], kernel_size=5, padding=2)

        self.pool_one_branch_four = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_one_branch_four = nn.LazyConv2d(out_channels=output_4, kernel_size=1)
    
    def forward(self, x):
        block_1 = self.ReLU(self.conv_one_branch_one(x))
        block_2 = self.ReLU(self.conv_two_branch_two(self.ReLU(self.conv_one_branch_two(x))))
        block_3 = self.ReLU(self.conv_two_branch_three(self.ReLU(self.conv_one_branch_three(x))))
        block_4 = self.ReLU(self.conv_one_branch_four(self.pool_one_branch_four(x)))
        return torch.cat((block_1, block_2, block_3, block_4), dim=1)
    

class GoogleLenet(nn.Module):
    """
    GoogLeNet-like architecture with Inception modules.

    This module is a variant of the GoogLeNet architecture, incorporating Inception blocks for deeper feature extraction.
    The model starts with several convolutional and max pooling layers, followed by three groups of Inception blocks.
    The output is flattened and passed through a final linear layer for classification.

    Args:
        input_channels (int): Number of input channels in the input data (e.g., 1 for grayscale images, 3 for RGB images).
        output_size (int): The number of classes for the output layer.

    Methods:
        forward(x, print_sizes=False):
            Defines the computation performed at every call. The input is passed through convolutional layers, 
            Inception blocks, and finally a linear layer for classification.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
                print_sizes (bool): If True, prints the size of the tensor after each major operation for debugging purposes.
            Returns:
                torch.Tensor: The output tensor representing the class logits.
    """
    def __init__(self, input_channels, output_size):
        super(GoogleLenet, self).__init__()

        self.ReLU = nn.ReLU()

        self.conv_one = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.pool_one = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv_two = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)

        self.conv_three = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1)
        self.pool_three = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_block_one = nn.ModuleList()
        self.inception_block_one.append(Inception(64, (96, 128), (16, 32), 32)) 
        self.inception_block_one.append(Inception(128, (128, 192), (32, 96), 64))
        self.inception_block_one_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_block_two = nn.ModuleList()
        self.inception_block_two.append(Inception(192, (96, 208), (16, 48), 64))
        self.inception_block_two.append(Inception(160, (112, 224), (24, 64), 64))
        self.inception_block_two.append(Inception(128, (128, 256), (24, 64), 64))
        self.inception_block_two.append(Inception(112, (144, 288), (32, 64), 64))
        self.inception_block_two.append(Inception(256, (160, 320), (32, 128), 128))
        self.inception_block_two_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_block_three = nn.ModuleList()
        self.inception_block_three.append(Inception(256, (160, 320), (32, 128), 128))
        self.inception_block_three.append(Inception(384, (192, 384), (48, 128), 128))
        self.inception_block_three_pool = nn.AdaptiveAvgPool2d((1,1))

        self.flatten = nn.Flatten()

        self.linear = nn.LazyLinear(output_size)

    def forward(self, x, print_sizes=False):
        layers = [
            ("conv_one", self.conv_one),
            ("pool_one", self.pool_one),
            ("conv_two", self.conv_two),
            ("conv_three", self.conv_three),
            ("pool_three", self.pool_three),
        ]
        
        # Passing through the initial convolutional and pooling layers
        for name, layer in layers:
            x = self.ReLU(layer(x))
            if print_sizes:
                print(f"After {name}: {x.shape}")

        # Inception block one
        for i, inception_block in enumerate(self.inception_block_one):
            x = inception_block(x)
            if print_sizes:
                print(f"After inception_block_one_{i}: {x.shape}")
        x = self.inception_block_one_pool(x)
        if print_sizes:
            print(f"After inception_block_one_pool: {x.shape}")

        # Inception block two
        for i, inception_block in enumerate(self.inception_block_two):
            x = inception_block(x)
            if print_sizes:
                print(f"After inception_block_two_{i}: {x.shape}")
        x = self.inception_block_two_pool(x)
        if print_sizes:
            print(f"After inception_block_two_pool: {x.shape}")

        # Inception block three
        for i, inception_block in enumerate(self.inception_block_three):
            x = inception_block(x)
            if print_sizes:
                print(f"After inception_block_three_{i}: {x.shape}")
        x = self.inception_block_three_pool(x)
        if print_sizes:
            print(f"After inception_block_three_pool: {x.shape}")

        # Flatten and pass through the final linear layer
        x = self.flatten(x)
        if print_sizes:
            print(f"After flatten: {x.shape}")

        x = self.linear(x)
        if print_sizes:
            print(f"After linear: {x.shape}")

        return x
    
def main():
    
    model = GoogleLenet(input_channels=1, output_size=4)
    x = torch.randn(1, 1, 96, 96)
    model(x, print_sizes=True)


if __name__ == "__main__":
    main()
