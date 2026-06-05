from scripts.dataset import GTZANDataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics as metrics

class GenreCNN(nn.Module):
    def __init__(self, num_classes=10):
        # Initialize the CNN architecture
        super(GenreCNN, self).__init__() # Call the parent class constructor
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.ReLU = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #Flatten the output from the convolutional layers and connect to fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 16 * 16, 256) 
        self.fc2 = nn.Linear(256, num_classes)

    
    # Define the forward pass of the CNN
    def forward(self, x):
        x = self.pool(self.ReLU(self.conv1(x)))
        x = self.pool(self.ReLU(self.conv2(x)))
        x = self.pool(self.ReLU(self.conv3(x)))
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.ReLU(self.fc1(x))
        x = self.fc2(x)
        return x
