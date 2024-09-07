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


class Residual_Block(nn.Module):
    """
    A residual block for a ResNet-style neural network.

    This block consists of two convolutional layers with batch normalization and ReLU activation. 
    Optionally, a 1x1 convolution can be applied to match the input and output dimensions for the residual connection.

    Args:
        input_channels (int): The number of input channels to the block.
        output_channels (int): The number of output channels for the convolutional layers.
        use_1x1conv (bool, optional): Whether to use a 1x1 convolution to match input and output dimensions for the skip connection. Default is False.
        strides (int, optional): The stride of the first convolution layer. Default is 1.

    Attributes:
        ReLU (nn.ReLU): ReLU activation function applied after each convolutional layer.
        conv_one (nn.Conv2d): First convolutional layer with a kernel size of 3.
        batch_norm_one (nn.BatchNorm2d): Batch normalization applied after the first convolutional layer.
        conv_two (nn.Conv2d): Second convolutional layer with a kernel size of 3.
        batch_norm_two (nn.BatchNorm2d): Batch normalization applied after the second convolutional layer.
        conv_three (nn.Conv2d or None): Optional 1x1 convolution applied when `use_1x1conv` is True, to adjust input dimensions.

    Methods:
        forward(x):
            Defines the forward pass of the residual block.
            
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).
            
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, output_channels, height, width) with the residual connection.
    """
    def __init__(self, input_channels, output_channels, use_1x1conv=False, strides=1, print_sizes=False):
        super().__init__()
        self.print_sizes = print_sizes

        self.ReLU = nn.ReLU()

        self.conv_one = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1, stride=strides)
        self.batch_norm_one = nn.BatchNorm2d(num_features=output_channels)
        
        self.conv_two = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1)
        self.batch_norm_two = nn.BatchNorm2d(num_features=output_channels)

        if use_1x1conv:
            self.conv_three = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, stride=strides)
        else:
            self.conv_three = None

    def forward(self, x):

        skip = x
        layers = [("block_conv_one", self.conv_one),
                  ("block_batch_norm_one", self.batch_norm_one),
                  ("block_ReLU_one", self.ReLU),
                  ("block_conv_two", self.conv_two),
                  ("block_batch_norm_two", self.batch_norm_two),
                  ]     
       
        if self.print_sizes:
            print()

        for name, layer in layers:
            x = layer(x)
            if self.print_sizes:
                details = f"    After {name}: {list(x.size())}"
            
                # Add details for each layer type
                if isinstance(layer, torch.nn.Conv2d):
                    details += f", Kernel Size: {layer.kernel_size}, Stride: {layer.stride}, Padding: {layer.padding}"
                elif isinstance(layer, torch.nn.BatchNorm2d):
                    details += f", Num Features: {layer.num_features}, Eps: {layer.eps}"
                elif isinstance(layer, torch.nn.Linear):
                    details += f", In Features: {layer.in_features}, Out Features: {layer.out_features}"
                
                print(details)  

        if self.conv_three:
            skip = self.conv_three(skip)
        
        output = self.ReLU(skip + x)
        if self.print_sizes:
            print(f"    After Final ReLU: {list(output.size())}")  
            print()

        return output
    
class ResNet(nn.Module):
    """
    A ResNet-style neural network.

    This model is composed of a series of residual blocks, followed by an adaptive average pooling layer 
    and a fully connected layer. The architecture of the network is determined by the `architecture` parameter.

    Args:
        architecture (list of tuples): A list where each tuple contains the number of residual blocks and 
                                       the number of output channels for each layer of blocks.
        input_channels (int): The number of input channels to the network (e.g., 3 for RGB images).
        output_size (int): The size of the output layer (e.g., the number of classes for classification).

    Attributes:
        ReLU (nn.ReLU): ReLU activation function applied after convolutional layers.
        conv_one (nn.Conv2d): The initial convolutional layer with a kernel size of 7.
        batch_norm_one (nn.BatchNorm2d): Batch normalization applied after the first convolutional layer.
        max_pool_one (nn.MaxPool2d): Max pooling layer applied after the first batch normalization.
        residual_blocks (nn.ModuleList): A list of residual blocks that form the main body of the network.
        adaptive_avg_pool (nn.AdaptiveAvgPool2d): Adaptive average pooling layer that outputs a 1x1 feature map.
        flatten (nn.Flatten): Flattening layer to convert 4D tensors to 2D before the fully connected layer.
        linear (nn.Linear): The fully connected layer that outputs to the final number of classes.

    Methods:
        block_maker(block_input_channels, num_residuals, output_channels, first_block=False):
            Helper function to create a sequence of residual blocks for each layer.

        forward(x, print_sizes=False):
            Defines the forward pass of the network.
            
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).
                print_sizes (bool, optional): If True, prints the size of the tensor after each layer. Default is False.
            
            Returns:
                torch.Tensor: Output tensor of the network representing class scores.
    """
    def __init__(self, architecture, input_channels, output_size, print_sizes=False, print_block_sizes=False):
        super(ResNet, self).__init__()

        self.print_sizes = print_sizes
        self.print_block_sizes = print_block_sizes

        self.ReLU = nn.ReLU()

        self.architecture = architecture

        self.conv_one = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.batch_norm_one = nn.BatchNorm2d(num_features=64)
        self.max_pool_one = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.residual_blocks = nn.ModuleList()

        input_channels = 64
        for idx, (num_residuals, output_channels) in enumerate(self.architecture):
            self.residual_blocks.append(self.block_maker(input_channels, num_residuals, output_channels, first_block=(idx==0)))
            input_channels = output_channels
        
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=output_channels, out_features=output_size)


    def block_maker(self, block_input_channels, num_residuals, output_channels, first_block=False):
        block = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                block.append(Residual_Block(block_input_channels, output_channels, use_1x1conv=True, strides=2, print_sizes=self.print_block_sizes))
            else:
                in_channels = block_input_channels if first_block else output_channels
                block.append(Residual_Block(in_channels, output_channels, print_sizes=self.print_block_sizes))
        return nn.Sequential(*block)
    
    def forward(self, x):

        layers = [("Conv_One", self.conv_one),
                  ("Batch_Norm_One", self.batch_norm_one),
                  ("ReLU_One", self.ReLU),
                  ("Max_Pool_One", self.max_pool_one)]
        
        layers.extend([(f"Residual_block_{i+1}", block) for i, block in enumerate(self.residual_blocks)])

        layers.extend([("Adaptive_Avg_Pool", self.adaptive_avg_pool), ("Flatten", self.flatten), ("Linear_layer", self.linear)])

        for name, layer in layers:
            x = layer(x)
            if self.print_sizes:
                details = f"After {name}: {list(x.size())}"
            
                # Add details for each layer type
                if isinstance(layer, torch.nn.Conv2d):
                    details += f", Kernel Size: {layer.kernel_size}, Stride: {layer.stride}, Padding: {layer.padding}"
                elif isinstance(layer, torch.nn.BatchNorm2d):
                    details += f", Num Features: {layer.num_features}, Eps: {layer.eps}"
                elif isinstance(layer, torch.nn.Linear):
                    details += f", In Features: {layer.in_features}, Out Features: {layer.out_features}"
                elif isinstance(layer, torch.nn.MaxPool2d):
                    details += f", Kernel Size: {layer.kernel_size}, Stride: {layer.stride}, Padding: {layer.padding}"
            
                print(details)
        return x
    
    

    

def main():

    x = torch.randn(1, 3, 224, 224)
    print(x.shape)
    arch = ((3, 64), (4, 128), (6, 256), (3, 512))
    model = ResNet(arch, 3, 10, print_sizes=False, print_block_sizes=False)
    x = model(x)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")
    


if __name__ == "__main__":
    main()