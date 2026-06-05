import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    
dataset=GTZANDataset('data/genres_original')
train_size=int(0.8*len(dataset))
test_size=len(dataset)-train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# Set the device to GPU if available, otherwise use CPU (AMD GPU support is the same 
#as CUDA in PyTorch, so it will automatically use the AMD GPU if it's available and
#  properly configured)
device = torch.device('cpu')
#Check if the computation device
print(f"Using device: {device}")
print(f"Training batches: {len(train_loader)}, Testing batches: {len(test_loader)}")

# Initialize the CNN model, loss function, and optimizer
model = GenreCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 30


# Train the CNN model
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0  # Initialize correct predictions counter
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs,1)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Batch Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}') 