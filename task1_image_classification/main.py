import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from keras.optimizers import Adam

def load_mnist_data():
    """Loads and preprocesses the MNIST dataset."""
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)

    # Reshape and normalize the dataset for CNNs (28x28 images)
    X = X.values.reshape(-1, 28, 28, 1)  # Reshape to (samples, 28, 28, 1) for CNN input
    X = X / 255.0  # Normalize pixel values to [0, 1]

    # Convert labels to one-hot encoding
    y = to_categorical(y, 10)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

class MnistClassifier:
    def __init__(self, algorithm='cnn'):
        self.algorithm = algorithm
        self.model = None

    def build_cnn(self):
        """Builds the CNN model."""
        self.model = Sequential()

        # Convolutional layer 1
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Convolutional layer 2
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Flatten the output for the fully connected layers
        self.model.add(Flatten())

        # Fully connected layer
        self.model.add(Dense(128, activation='relu'))

        # Output layer with 10 classes (for 10 digits)
        self.model.add(Dense(10, activation='softmax'))

        # Compile the model
        self.model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train):
        """Trains the selected model."""
        if self.algorithm == 'cnn':
            print("Training CNN model...")
            self.build_cnn()
            self.model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

    def predict(self, X_test):
        """Makes predictions using the trained model."""
        return np.argmax(self.model.predict(X_test), axis=1)

def main():
    """Runs model training and testing."""
    X_train, X_test, y_train, y_test = load_mnist_data()

    # Ask user for the model type, but only allow 'cnn' now
    model_choice = input("Enter model type (cnn): ").strip().lower()

    if model_choice != 'cnn':
        print("Currently, only CNN model type is supported.")
        return

    # Initialize and train the classifier
    classifier = MnistClassifier(algorithm=model_choice)
    classifier.train(X_train, y_train)

    # Make predictions
    predictions = classifier.predict(X_test)

    # Evaluate the model
    accuracy = np.mean(np.argmax(y_test, axis=1) == predictions) * 100
    print(f"\nModel Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()

