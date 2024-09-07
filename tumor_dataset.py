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


class Tumor_Dataset(Dataset):
    """
    A custom PyTorch Dataset for loading tumor images and their corresponding labels.

    This dataset class is designed to load images from specified folders, resize them, 
    convert them to grayscale, and prepare them as tensors for use in a neural network model.
    The images are labeled based on the folder they are contained in.

    Args:
        img_size (tuple): The target size of the images as (width, height).
        folders (dict): A dictionary where keys are folder paths and values are the 
                        corresponding labels for the images in those folders.

    Attributes:
        folders (dict): Stores the input dictionary mapping folder paths to labels.
        image_size (tuple): The target size of the images as (width, height).
        file_paths (list): List of file paths for all images in the dataset.
        labels (list): List of labels corresponding to each image in `file_paths`.

    Methods:
        prepare_filepaths(folders):
            Populates the `file_paths` and `labels` attributes by iterating over the provided folders.

        load_image(image_path):
            Loads an image from the given path, converts it to grayscale, resizes it to `image_size`, 
            and returns it as a tensor.

        __getitem__(idx):
            Retrieves the image and label at the specified index `idx`.
            
            Args:
                idx (int): Index of the image and label to retrieve.
                
            Returns:
                tuple: A tuple containing the image tensor and its corresponding label.

        __len__():
            Returns the total number of images in the dataset.
            
            Returns:
                int: The number of images in the dataset.
    """
    def __init__(self, image_size, folders):
        self.folders = folders
        self.image_size = image_size
        
        self.file_paths = []
        self.labels = []

        self.prepare_filepaths(self.folders)

    def prepare_filepaths(self, folders):
        for folder, label in folders.items():
            for filename in os.listdir(folder):
                self.file_paths.append(os.path.join(folder, filename))
                self.labels.append(label)
    
    def load_image(self, image_path):

        img = Image.open(image_path).convert('L')
        img = img.resize(self.image_size)
        img = np.array(img, dtype=np.float32)
        img = torch.tensor(img, dtype=torch.float32)
        img = img.unsqueeze(0)
        return img
       
    def __getitem__(self, idx):

        image_path = self.file_paths[idx]
        image = self.load_image(image_path)
        label = self.labels[idx]

        return image, label

    def __len__(self):
        return len(self.file_paths)
