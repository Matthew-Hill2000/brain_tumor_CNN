import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import random
from PIL import Image
import os
from tumor_dataset import Tumor_Dataset 


class Trainer():
    """
    A training pipeline for a neural network model using PyTorch.

    This class encapsulates the process of loading data, initializing a model,
    training the model, and evaluating its performance on a test set. The
    model used is a LeNet-based architecture, and the dataset is expected to be
    a custom tumor dataset.

    Args:
        model_class (type): The class of the neural network model to be used. It should be a subclass of `nn.Module`.
        model_kwargs (dict): Keyword arguments required to initialize the model class.
        dataset_class (type): The class of the dataset to be used. It should be a subclass of `torch.utils.data.Dataset`.
        dataset_kwargs (dict): Keyword arguments required to initialize the dataset class.
        learning_rate (float): Learning rate for the optimizer.
        num_epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size for data loading.
        train_folders (dict): Dictionary mapping training folder paths to their respective labels.
        test_folders (dict): Dictionary mapping testing folder paths to their respective labels.
        validation_split (float): Proportion of the training data to be used for validation.

    Attributes:
        device (torch.device): The device on which the model will be trained (CPU or GPU).
        learning_rate (float): Learning rate for the optimizer.
        num_epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size for data loading.
        validation_split (float): Proportion of the training data to be used for validation.
        criterion (nn.Module): Loss function used for training.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        test_loader (DataLoader): DataLoader for the testing dataset.
        model_class (type): The class of the neural network model to be used. It should be a subclass of `nn.Module`.
        model (nn.Module): The neural network model to be trained and tested.
        dataset_class (type): The class of the dataset to be used. It should be a subclass of `torch.utils.data.Dataset`.
        dataset_kwargs (dict): Keyword arguments for dataset initialization, including 'train_folders' and 'test_folders'.

    Methods:
        load_data():
            Loads the training and testing datasets, initializes DataLoader objects for each.

        load_model():
            Initializes the model, loss function, and optimizer. Moves the model to the appropriate device.

        train():
            Trains the model for the specified number of epochs. Prints the average training loss per epoch.

        test():
            Evaluates the model on the test dataset. Returns the accuracy, as well as dictionaries of correctly
            and incorrectly classified samples per class.

            Returns:
                tuple: A tuple containing:
                    - acc (float): The overall accuracy of the model on the test set.
                    - correct_classified (dict): A dictionary with the number of correctly classified samples per class.
                    - incorrect_classified (dict): A dictionary with the number of incorrectly classified samples per class.
    """
    def __init__(self, model_class, model_kwargs, dataset_class, dataset_kwargs, learning_rate, num_epochs, batch_size, train_folders, test_folders, validation_split=0.1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.criterion = None
        self.optimizer = None
        self.train_loader = None
        self.test_loader = None
        
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.model = None

        self.dataset_class = dataset_class
        self.dataset_kwargs = dataset_kwargs

        self.train_folders = train_folders
        self.test_folders = test_folders

    def load_data(self):

        # Extract folders from dataset_kwargs
        train_folders = self.dataset_kwargs.pop('train_folders')
        test_folders = self.dataset_kwargs.pop('test_folders')

        # Initialize datasets with the provided class and arguments
        full_dataset = self.dataset_class(folders=train_folders, **self.dataset_kwargs)
        
        # Split dataset into training and validation sets
        total_size = len(full_dataset)
        val_size = int(self.validation_split * total_size)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, num_workers=1, shuffle=True)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, num_workers=1, shuffle=False)
        
        test_dataset = self.dataset_class(folders=test_folders, **self.dataset_kwargs)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, num_workers=1, shuffle=False) 

    def load_model(self):
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Selected device: {self.device}")    
        
        self.model = self.model_class(**self.model_kwargs).to(self.device)
        self.criterion =  nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        is_on_gpu = next(self.model.parameters()).is_cuda
        print(f"Model is on GPU:{is_on_gpu}")

    def train(self):
        self.model.train()
        num_batches_train = len(self.train_loader)
        num_batches_val = len(self.val_loader)
        
        train_losses = []
        val_losses = []

        for epoch in range(self.num_epochs):
            print(f"Epoch: {epoch}")

            # Training phase
            train_loss = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            average_train_loss = train_loss / num_batches_train
            train_losses.append(average_train_loss)
            print(f"Average train_loss: {average_train_loss:.4f}")

            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

            average_val_loss = val_loss / num_batches_val
            val_losses.append(average_val_loss)
            print(f"Average val_loss: {average_val_loss:.4f}")

            self.model.train()

        # Plotting the losses
        plt.figure()
        plt.plot(range(self.num_epochs), train_losses, label='Training Loss')
        plt.plot(range(self.num_epochs), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss over Epochs')
        plt.legend()
        plt.show()

    def test(self):
        self.model.eval()

        n_samples = 0
        correct_classified = {class_idx:0 for class_idx in range(4)}
        incorrect_classified = {class_idx:0 for class_idx in range(4)}

        for images, targets in self.test_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = nn.functional.softmax(self.model(images), dim=1)
            _, predicted = torch.max(outputs, 1)
            n_samples += targets.size(0)

            for index in range(len(predicted)):
                    if predicted[index] == targets[index]:
                        correct_classified[targets[index].item()] += 1
                    else:
                        incorrect_classified[targets[index].item()] += 1

        acc = 100.0 * sum(correct_classified.values()) / n_samples
        return acc, correct_classified, incorrect_classified
        

    