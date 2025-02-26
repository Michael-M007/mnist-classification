from models.random_forest import RandomForestMnist
from models.feed_forward_nn import FeedForwardNN
from models.cnn import CNN

class MnistClassifier:
    def __init__(self, algorithm):
        if algorithm == 'rf':
            self.model = RandomForestMnist()
        elif algorithm == 'nn':
            self.model = FeedForwardNN()
        elif algorithm == 'cnn':
            self.model = CNN()
        else:
            raise ValueError("Invalid algorithm. Choose 'rf', 'nn', or 'cnn'.")

    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
