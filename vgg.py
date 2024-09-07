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

class VGG_Block(nn.Module):
    """
    A convolutional block for the VGG network.

    This block consists of a sequence of convolutional layers followed by ReLU activations
    and a MaxPooling layer. The number of convolutional layers is determined by `num_convs`.

    Args:
        num_convs (int): The number of convolutional layers in this block.
        input_channels (int): The number of input channels to the first convolutional layer.
        output_channels (int): The number of output channels from each convolutional layer in this block.

    Attributes:
        ReLU (nn.ReLU): ReLU activation function applied after each convolutional layer.
        layers (list): A list that contains the convolutional layers, ReLU activations, and the MaxPooling layer.
        block (nn.Sequential): A sequential container that encapsulates the entire block.

    Methods:
        forward(x):
            Passes the input tensor through the block of convolutional layers, ReLU activations, and MaxPooling.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).
            
            Returns:
                torch.Tensor: Output tensor of the network representing class scores.
    
    """
    def __init__(self, num_convs, input_channels, output_channels):
        super(VGG_Block, self).__init__()

        self.ReLU = nn.ReLU()

        self.layers = []
        for _ in range(num_convs):
            self.layers.append(nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1))
            self.layers.append(nn.ReLU())
            # Update input_channels for the next layer to match output_channels
            input_channels = output_channels
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.block = nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.block(x)

class VGG(nn.Module):
    """
    A VGG-style neural network.

    This model consists of several VGG_Block modules, followed by fully connected layers. 
    The architecture of the network is defined by the `architecture` parameter.

    Args:
        architecture (list of tuples): A list where each tuple contains the number of convolutional layers 
                                       and the number of output channels for each VGG block.
        input_channels (int): The number of input channels to the network (e.g., 1 for grayscale images, 3 for RGB).
        output_size (int): The size of the output layer (e.g., the number of classes for classification).

    Attributes:
        ReLU (nn.ReLU): ReLU activation function applied after each fully connected layer.
        convolutional_blocks (nn.ModuleList): A list of VGG_Block modules.
        flatten (nn.Flatten): A flattening layer to convert 4D tensors to 2D before fully connected layers.
        fully_connected_one (nn.LazyLinear): The first fully connected layer with 4096 units.
        dropout_one (nn.Dropout): Dropout applied after the first fully connected layer.
        fully_connected_two (nn.LazyLinear): The second fully connected layer with 4096 units.
        dropout_two (nn.Dropout): Dropout applied after the second fully connected layer.
        fully_connected_three (nn.LazyLinear): The third fully connected layer that outputs to the final number of classes.

    Methods:
        forward(x, print_sizes=False):
            Defines the forward pass of the network.
            
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).
                print_sizes (bool): If True, prints the size of the tensor after each layer.

            Returns:
                torch.Tensor: Output tensor of the network representing class scores.
    """
    def __init__(self, architecture, input_channels, output_size):
        super(VGG, self).__init__()

        self.ReLU = nn.ReLU()

        # Use nn.ModuleList to store the blocks
        self.convolutional_blocks = nn.ModuleList()
        
        for num_convs, out_channels in architecture:
            self.convolutional_blocks.append(VGG_Block(num_convs, input_channels, out_channels))
            input_channels = out_channels

        self.flatten = nn.Flatten()
        self.fully_connected_one = nn.LazyLinear(out_features=4096)
        self.dropout_one = nn.Dropout(p=0.5)
        self.fully_connected_two = nn.LazyLinear(out_features=4096)
        self.dropout_two = nn.Dropout(p=0.5)
        self.fully_connected_three = nn.LazyLinear(out_features=output_size)

    def forward(self, x, print_sizes=False):
        layers = []

        # Add convolutional blocks to the layers list
        layers.extend([(f"VGG_block_{i+1}", block) for i, block in enumerate(self.convolutional_blocks)])

        # Add fully connected layers and other layers
        layers.extend([
            ("flatten", self.flatten),
            ("fully_connected_one", self.fully_connected_one),
            ("ReLU", self.ReLU),
            ("dropout_one", self.dropout_one),
            ("fully_connected_two", self.fully_connected_two),
            ("ReLU", self.ReLU),
            ("dropout_two", self.dropout_two),
            ("fully_connected_three", self.fully_connected_three)
        ])

        for name, layer in layers:
            x = layer(x)
            if print_sizes:
                print(f"After {name}: {list(x.size())}")

        return x
    
def main():
    arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    model = VGG(architecture=arch, input_channels=1, output_size=10)
    x = torch.randn(1, 1, 256, 256)
    model(x, print_sizes=True)


if __name__ == "__main__":
    main()





