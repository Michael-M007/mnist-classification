import torch
import torch.nn as nn
import torch.optim as optim
from mnist_classifier_interface import MnistClassifierInterface

class FeedForwardNN(nn.Module, MnistClassifierInterface):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

    def train(self, X_train, y_train):
        # Implement PyTorch training loop
        pass

    def predict(self, X_test):
        # Implement PyTorch prediction logic
        pass
