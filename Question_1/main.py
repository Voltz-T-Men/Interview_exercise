from utils import MnistDataloader
import numpy as np
from model import MulticlassSVM

def main():
    """
    Main function to load the MNIST dataset, train a Multiclass SVM model, and evaluate its accuracy.
    
    This function performs the following steps:
    1. Initializes the data loader with file paths for MNIST training and test datasets.
    2. Loads the training and test data.
    3. Uses a subset of the data for simplicity.
    4. Trains a Multiclass SVM model on the subset of training data.
    5. Makes predictions on the subset of test data.
    6. Evaluates the model by computing its accuracy on the test data and prints the result.
    """

    # File paths for the MNIST dataset
    training_images_filepath = "mnist_dataset/train-images.idx3-ubyte"
    training_labels_filepath = "mnist_dataset/train-labels.idx1-ubyte"
    test_images_filepath = "mnist_dataset/t10k-images.idx3-ubyte"
    test_labels_filepath = "mnist_dataset/t10k-labels.idx1-ubyte"
    
    # Initialize the MnistDataloader
    Mnist = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    
    # Load the MNIST dataset
    train_images, train_labels, test_images, test_labels = Mnist.load_data()

    # For simplicity, use only a subset of the data
    train_images_subset = train_images[:100]
    train_labels_subset = train_labels[:100]
    test_images_subset = test_images[:3]
    test_labels_subset = test_labels[:3]

    # Initialize the MulticlassSVM model
    msvm = MulticlassSVM(learning_rate=0.001, lambda_param=0.01, n_iters=100)
    
    # Train the SVM model on the training subset
    msvm.fit(train_images_subset, train_labels_subset)

    # Make predictions on the test subset
    predictions = msvm.predict(test_images_subset)

    # Evaluate the model by calculating accuracy
    
    # print(predictions)
    # print(test_labels_subset)
    accuracy = np.mean(predictions == test_labels_subset)
    print(f'Multiclass SVM Accuracy: {accuracy:.2f}')

if __name__ == "__main__":
    main()
