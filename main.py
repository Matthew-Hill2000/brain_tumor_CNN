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
from models.lenet import LeNet
from tumor_dataset import Tumor_Dataset 
from tumor_detection import Trainer
from models.alexnet import AlexNet
from models.vgg import VGG
from models.network_in_network import NiN
from models.resnet import ResNet


def main():
    LEARNING_RATE = 0.1
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    MODEL = ResNet
    model_kwargs = {'architecture': ((3, 64), (4, 128), (6, 256), (3, 512)), 
                    'input_channels': 1, 
                    'output_size': 4}


    TRAIN_FOLDERS = {"Training/glioma":0,
                "Training/meningioma":1, 
                "Training/notumor":2,
                "Training/pituitary":3}
    
    TEST_FOLDERS = {"Testing/glioma":0,
                "Testing/meningioma":1, 
                "Testing/notumor":2,
                "Testing/pituitary":3}

    dataset_class = Tumor_Dataset
    dataset_kwargs = {'image_size': (256, 256),
                      'train_folders': TRAIN_FOLDERS,
                      'test_folders': TEST_FOLDERS
    }

    
    classification_model = Trainer(
                                model_class=MODEL,
                                model_kwargs=model_kwargs,
                                dataset_class=dataset_class,
                                dataset_kwargs=dataset_kwargs,
                                learning_rate=LEARNING_RATE, 
                                num_epochs=NUM_EPOCHS, 
                                batch_size=BATCH_SIZE, 
                                train_folders=TRAIN_FOLDERS, 
                                test_folders=TEST_FOLDERS
                                )
    
    classification_model.load_data() 
    classification_model.load_model()
    classification_model.train()
    acc, correct_classified, incorrect_classified = classification_model.test()

    print(f"Accuracy: {acc}")
    print(f"Correct Classified: {correct_classified}")
    print(f"Incorrect Classified: {incorrect_classified}")

if __name__ == "__main__":
    main()