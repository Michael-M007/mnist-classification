from abc import ABC, abstractmethod
import numpy as np

class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the model on the given dataset."""
        pass

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict the labels for the given test data."""
        pass
