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


class LeNet(nn.Module):
    """
    A PyTorch implementation of the LeNet-5 architecture.

    This class defines the LeNet-5 model, a convolutional neural network designed
    for image classification tasks. The network consists of two convolutional layers 
    followed by two fully connected layers, and a final output layer. Each convolutional 
    layer is followed by a ReLU activation and a max-pooling operation. The fully 
    connected layers also use ReLU activations.

    Attributes:
        conv_one (nn.Conv2d): First convolutional layer with 6 output channels.
        pool_one (nn.MaxPool2d): First max-pooling layer with a 2x2 kernel.
        conv_two (nn.Conv2d): Second convolutional layer with 16 output channels.
        pool_two (nn.MaxPool2d): Second max-pooling layer with a 2x2 kernel.
        flatten (nn.Flatten): Layer to flatten the input for the fully connected layers.
        fully_connected_one (nn.LazyLinear): First fully connected layer with 120 units.
        fully_connected_two (nn.Linear): Second fully connected layer with 84 units.
        fully_connected_three (nn.Linear): Final fully connected layer with `output_size` units.
        ReLU (nn.ReLU): ReLU activation function used after each convolutional and fully connected layer.

    Args:
        input_channels (int): Number of channels in the input image.
        output_size (int): Number of classes for the classification task.

    Methods:
        forward(x, print_sizes=False):
            Defines the forward pass of the network.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).
                print_sizes (bool): If True, prints the size of the tensor after each layer.
            Returns:
                torch.Tensor: Output tensor of the network representing class scores.
    """
    def __init__(self, input_channels, output_size):
        super(LeNet, self).__init__()

        self.ReLU = nn.ReLU()

        self.conv_one = nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=5, padding=2)
        self.pool_one = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_two = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool_two = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        self.fully_connected_one = nn.LazyLinear(out_features=120)
        self.fully_connected_two = nn.Linear(in_features=120, out_features=84)
        self.fully_connected_three = nn.Linear(in_features=84, out_features=output_size)

    def forward(self, x, print_sizes=False):

        layers = [
            ("conv_one", self.conv_one),
            ("ReLU", self.ReLU),
            ("pool_one", self.pool_one),
            ("conv_two", self.conv_two),
            ("ReLU", self.ReLU),
            ("pool_two", self.pool_two),
            ("flatten", self.flatten),
            ("fully_connected_one", self.fully_connected_one),
            ("ReLU", self.ReLU),
            ("fully_connected_two", self.fully_connected_two),
            ("ReLU", self.ReLU),
            ("fully_connected_three", self.fully_connected_three),
        ]

        
        for name, layer in layers:
            x = layer(x)
            if print_sizes:
                print(f"After {name}: {list(x.size())}")

        return x
    
    
    