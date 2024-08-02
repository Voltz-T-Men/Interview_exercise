import numpy as np
from model import train
from utils import *

def load_mnist_data():
    """
    Load the MNIST dataset from raw binary files.

    Returns:
    --------
    tuple of np.ndarray
        A tuple containing four numpy arrays:
        - X_train: Training images.
        - y_train: Training labels.
        - X_test: Test images.
        - y_test: Test labels.
    """
    training_images_filepath = "mnist_dataset/train-images.idx3-ubyte"
    training_labels_filepath = "mnist_dataset/train-labels.idx1-ubyte"
    test_images_filepath = "mnist_dataset/t10k-images.idx3-ubyte"
    test_labels_filepath = "mnist_dataset/t10k-labels.idx1-ubyte"
    mnist = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    return mnist.load_data()

def inference(model, X_train, y_train, input):
    """
    Perform inference to predict the label of an input sample.

    Parameters:
    -----------
    model : NeuralNetwork
        Trained neural network model.
    X_train : np.ndarray
        Training images.
    y_train : np.ndarray
        Training labels.
    input : np.ndarray
        Input sample for which to predict the label.

    Returns:
    --------
    int
        Predicted label for the input sample.
    """
    label_features_list, unique_labels = extract_label_features(model, X_train, y_train)
    input_features = model.forward(input)
    similarities = [cosine_similarity(input_features, label_feature) for label_feature in label_features_list]
    predicted_label = unique_labels[np.argmax(similarities)]
    return predicted_label

def main():
    """
    Main function to load data, train the model, and perform inference on a test sample.
    """
    X_train, y_train, X_test, y_test = load_mnist_data()
    X_train = X_train[:5000]  # Limiting to a smaller subset for faster training
    y_train = y_train[:5000]
    model = train(X_train, y_train)
    input_sample = X_test[0]
    predicted_label = inference(model, X_train, y_train, input_sample)
    print(f"Predicted Label: {predicted_label}, True Label: {y_test[0]}")

if __name__ == "__main__":
    main()
