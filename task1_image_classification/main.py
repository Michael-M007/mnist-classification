import numpy as np
from mnist_classifier import MnistClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_mnist_data():
    """Loads and preprocesses the MNIST dataset."""
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)

    # Normalize the dataset for neural networks
    X = X / 255.0

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def main():
    """Runs model training and testing."""
    X_train, X_test, y_train, y_test = load_mnist_data()

    # Ask user for the model type
    model_choice = input("Enter model type (rf, nn, cnn): ").strip().lower()

    if model_choice not in ['rf', 'nn', 'cnn']:
        print("Invalid model choice. Please select 'rf', 'nn', or 'cnn'.")
        return

    # Initialize and train the classifier
    classifier = MnistClassifier(algorithm=model_choice)
    print(f"\nTraining the {model_choice.upper()} model...")
    classifier.train(X_train, y_train)

    # Make predictions
    predictions = classifier.predict(X_test)

    # Evaluate the model
    accuracy = np.mean(predictions == y_test) * 100
    print(f"\nModel Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
