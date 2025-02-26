import torch.nn as nn
import torch
from mnist_classifier_interface import MnistClassifierInterface

class CNN(nn.Module, MnistClassifierInterface):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64*5*5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def train(self, X_train, y_train):
        # Implement PyTorch training loop
        pass

    def predict(self, X_test):
        # Implement PyTorch prediction logic
        pass
