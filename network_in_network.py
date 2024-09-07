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

class NiN_Block(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding):
        super(NiN_Block, self).__init__()

        self.ReLU = nn.ReLU()

        self.conv_one = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_two = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=1)
        self.conv_three = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=1) 

    def forward(self, x):
        x = self.ReLU(self.conv_one(x))
        x = self.ReLU(self.conv_two(x))
        x = self.ReLU(self.conv_three(x))
        return x

class NiN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(NiN, self).__init__()

        self.nin_block_one = NiN_Block(input_channels=input_channels, output_channels=96, kernel_size=11, stride=4, padding=0)
        self.pool_one = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.nin_block_two = NiN_Block(input_channels=96, output_channels=256, kernel_size=5, stride=1, padding=2)
        self.pool_two = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.nin_block_three = NiN_Block(input_channels=256, output_channels=384, kernel_size=3, stride=1, padding=1)
        self.pool_three = nn.MaxPool2d(kernel_size=3, stride=2)

        self.dropout = nn.Dropout(p=0.5)

        self.nin_block_four = NiN_Block(input_channels=384, output_channels=output_size, kernel_size=3, stride=1, padding=1)
        self.pool_four = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()

    def forward(self, x, print_sizes=False):
        layers = [
            ("nin_block_one", self.nin_block_one),
            ("pool_one", self.pool_one),
            ("nin_block_two", self.nin_block_two),
            ("pool_two", self.pool_two),
            ("nin_block_three", self.nin_block_three),
            ("pool_three", self.pool_three),
            ("dropout", self.dropout),
            ("nin_block_four", self.nin_block_four),
            ("pool_four", self.pool_four),
            ("flatten", self.flatten)
        ]

        for name, layer in layers:
            x = layer(x)
            if print_sizes:
                print(f"After {name}: {list(x.size())}")

        return x
    

def main():
    arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    model = NiN(input_channels=1, output_size=4)
    x = torch.randn(1, 1, 224, 224)
    model(x, print_sizes=True)


if __name__ == "__main__":
    main()
