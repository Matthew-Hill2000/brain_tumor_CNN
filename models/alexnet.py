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

class AlexNet(nn.Module):
    """
    A PyTorch implementation of the AlexNet architecture.

    This class defines the AlexNet model, a convolutional neural network designed
    for image classification tasks. The network consists of five convolutional 
    layers followed by two fully connected layers and a final output layer. The
    first two convolutions are followed by a ReLU activation function and a
    max-pooling operation. The following two convolutions are followed only by
    a ReLU activation, and the final convolution is followed again by a ReLU 
    activation function and a max-pooling operation. The fully connected layers
    also use ReLU activations and include dropout for regularization.

    Args:
        input_channels (int): Number of channels in the input image.
        output_size (int): Number of classes for the classification task.

    Attributes:
        conv_one (nn.Conv2d): First convolutional layer with 96 output channels.
        pool_one (nn.MaxPool2d): First max-pooling layer with a 3x3 kernel and a stride of 2.
        conv_two (nn.Conv2d): Second convolutional layer with 256 output channels.
        pool_two (nn.MaxPool2d): Second max-pooling layer with a 3x3 kernel and a stride of 2.
        conv_three (nn.Conv2d): Third convolutional layer with 384 output channels.
        conv_four (nn.Conv2d): Fourth convolutional layer with 384 output channels.
        conv_five (nn.Conv2d): Fifth convolutional layer with 256 output channels.
        pool_five (nn.MaxPool2d): Final max-pooling layer with a 3x3 kernel and a stride of 2.
        flatten (nn.Flatten): Layer to flatten the input for the fully connected layers.
        fully_connected_one (nn.LazyLinear): First fully connected layer with 4096 units.
        dropout_one (nn.Dropout): Dropout layer with a probability of 0.5 applied after the first fully connected layer.
        fully_connected_two (nn.LazyLinear): Second fully connected layer with 4096 units.
        dropout_two (nn.Dropout): Dropout layer with a probability of 0.5 applied after the second fully connected layer.
        fully_connected_three (nn.LazyLinear): Final fully connected layer with `output_size` units.

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
        super(AlexNet, self).__init__()

        self.ReLU = nn.ReLU()

        self.conv_one = nn.Conv2d(in_channels=input_channels, out_channels=96, kernel_size=11, stride=4)
        self.pool_one = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv_two = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.pool_two = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv_three = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv_four = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv_five = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.pool_five = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten = nn.Flatten()

        self.fully_connected_one = nn.LazyLinear(out_features=4096)
        self.dropout_one = nn.Dropout(p=0.5)
        self.fully_connected_two = nn.LazyLinear(out_features=4096)
        self.dropout_two = nn.Dropout(p=0.5)
        self.fully_connected_three = nn.LazyLinear(out_features=output_size)

    def forward(self, x, print_sizes=False):

        layers = [
            ("conv_one", self.conv_one),
            ("ReLU", self.ReLU),
            ("max-pool_one", self.pool_one),
            ("conv_two", self.conv_two),
            ("ReLU", self.ReLU),
            ("max-pool_two", self.pool_two),
            ("conv_three", self.conv_three),
            ("ReLU", self.ReLU),
            ("conv_four", self.conv_four),
            ("ReLU", self.ReLU),
            ("conv_five", self.conv_five),
            ("ReLU", self.ReLU),
            ("max-pool_five", self.pool_five),
            ("flatten", self.flatten),
            ("fully_connected_one", self.fully_connected_one),
            ("ReLU", self.ReLU),
            ("Dropout_one", self.dropout_one),
            ("fully_connected_two", self.fully_connected_two),
            ("ReLU", self.ReLU),
            ("Dropout_two", self.dropout_two),
            ("fully_connected_three", self.fully_connected_three),
        ]

        for name, layer in layers:
            x = layer(x)
            if print_sizes:
                print(f"After {name}: {list(x.size())}")

        return x
    

def main():
     test_array = torch.randn(1, 3, 224, 224)
     model = AlexNet(input_channels=3, output_size=4)

     output = model(test_array, print_sizes=True)

     print(f"output shape: {output.shape}")
    
if __name__ == "__main__":
    main()