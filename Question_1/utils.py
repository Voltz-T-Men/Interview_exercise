import struct
import numpy as np

class MnistDataloader:
    """
    A class to load the MNIST dataset from raw binary files.

    Attributes:
    -----------
    training_images_filepath : str
        Path to the training images file.
    training_labels_filepath : str
        Path to the training labels file.
    test_images_filepath : str
        Path to the test images file.
    test_labels_filepath : str
        Path to the test labels file.

    Methods:
    --------
    read_images_labels(images_filepath, labels_filepath):
        Reads images and labels from the given file paths.
    load_data():
        Loads and returns the training and test datasets.
    """

    def __init__(self, training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath):
        """
        Initializes the MnistDataloader with file paths for training and test datasets.

        Parameters:
        -----------
        training_images_filepath : str
            Path to the training images file.
        training_labels_filepath : str
            Path to the training labels file.
        test_images_filepath : str
            Path to the test images file.
        test_labels_filepath : str
            Path to the test labels file.
        """
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        """
        Reads images and labels from the given file paths.

        Parameters:
        -----------
        images_filepath : str
            Path to the images file.
        labels_filepath : str
            Path to the labels file.

        Returns:
        --------
        tuple of np.ndarray
            A tuple containing two numpy arrays:
            - images: An array of shape (num_images, rows*cols) containing the pixel values.
            - labels: An array of shape (num_labels,) containing the labels.
        """
        # Read labels
        with open(labels_filepath, 'rb') as label_file:
            magic, num_labels = struct.unpack(">II", label_file.read(8))
            labels = np.fromfile(label_file, dtype=np.uint8)

        # Read images
        with open(images_filepath, 'rb') as image_file:
            magic, num_images, rows, cols = struct.unpack(">IIII", image_file.read(16))
            images = np.fromfile(image_file, dtype=np.uint8).reshape(num_images, rows * cols)

        return images, labels

    def load_data(self):
        """
        Loads and returns the training and test datasets.

        Returns:
        --------
        tuple of np.ndarray
            A tuple containing four numpy arrays:
            - x_train: Training images.
            - y_train: Training labels.
            - x_test: Test images.
            - y_test: Test labels.
        """
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return x_train, y_train, x_test, y_test
