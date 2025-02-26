# Project Title - MNIST Image Classification with OOP

This project implements three different classification models for the MNIST dataset using an object-oriented approach in Python. The models include:

1. **Random Forest (RF)**
2. **Feed-Forward Neural Network (NN)**
3. **Convolutional Neural Network (CNN)**
Each model is implemented as a separate class that follows a common interface (`MnistClassifierInterface`).
A unified wrapper class (`MnistClassifier`) is also provided to allow seamless switching between models.

## Built With
- Python
- MIniconda
- VS Code
- MyCharm
- Githab
- Jupyter
## Installation & Setup
### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/mnist-classification.git
cd mnist-classification/task1_image_classification
### **2. Set Up a Virtual Environment**
python -m venv venv
venv\Scripts\activate  # Windows
### **2. Install Dependencies**
pip install -r requirements.txt

## Project Structure
task1_image_classification/
│── mnist_classifier_interface.py    # Interface defining train & predict methods
│── models/
│   ├── random_forest.py             # Random Forest implementation
│   ├── feed_forward_nn.py           # Feed-Forward Neural Network
│   ├── cnn.py                       # Convolutional Neural Network
│── mnist_classifier.py              # Wrapper class for selecting a model
│── main.py                          # Script to train & test models
│── requirements.txt                  # Dependencies list
│── README.md                        # Documentation
│── demo_notebook.ipynb               # Jupyter Notebook for demonstration

## Usage
1. Training a Model
Modify and run main.py:

from mnist_classifier import MnistClassifier
from data_loader import load_mnist_data  # Assume you have a function to load data

# Load dataset
X_train, y_train, X_test, y_test = load_mnist_data()

# Choose a model ('rf', 'nn', or 'cnn')
classifier = MnistClassifier(algorithm='rf')

# Train the model
classifier.train(X_train, y_train)

# Make predictions
predictions = classifier.predict(X_test)
2. Running the Demo Notebook
jupyter notebook demo_notebook.ipynb

## Edge Cases Considered
1. The CNN expects image tensors, while RF and NN expect flattened arrays. Proper preprocessing is done accordingly.
2. Invalid Algorithm Selection so If an invalid model type is selected, an error is raised.
3. The notebook has an example where only a subset of MNIST is used to test model adaptability.

## Contact
Michael Remeta - remeta16bk@gmail.com
