import struct
import numpy as np

def triplet_loss(anchor, positive, negative, margin=0.1):
    """
    Computes the Triplet Loss for given anchor, positive, and negative samples.

    Parameters:
    - anchor: np.ndarray
      Feature vector of the anchor sample.
    - positive: np.ndarray
      Feature vector of the positive sample (same class as anchor).
    - negative: np.ndarray
      Feature vector of the negative sample (different class from anchor).
    - margin: float, default=1.0
      Margin for the loss function to ensure the distance between anchor and negative is greater than the distance 
      between anchor and positive by at least the margin.

    Returns:
    - total_loss: float
      The calculated triplet loss value.
    """
    # Compute the squared distance between the anchor and the positive example
    pos_dist = np.sum(np.square(anchor - positive), axis=-1)
    
    # Compute the squared distance between the anchor and the negative example
    neg_dist = np.sum(np.square(anchor - negative), axis=-1)
    
    # Compute the Triplet Loss using the margin
    loss = np.maximum(0, pos_dist - neg_dist + margin)
    
    return loss

def create_triplets(X, y, batch_size):
    """
    Generate triplets (anchor, positive, negative) for training.

    Parameters:
    - X: np.ndarray
      Feature vectors of the samples.
    - y: np.ndarray
      Labels of the samples.
    - batch_size: int
      Number of triplets to generate.

    Returns:
    - anchor: np.ndarray
      Array of anchor samples.
    - positive: np.ndarray
      Array of positive samples corresponding to anchors.
    - negative: np.ndarray
      Array of negative samples corresponding to anchors.
    """
    anchor, positive, negative = [], [], []
    for _ in range(batch_size):
        # Select a random anchor sample
        idx = np.random.randint(0, len(X)) #idx choose a index in sample vector x and add it to anchor list
        anchor.append(X[idx])
        
        
        # Select a positive sample (different sample of the same class as the anchor)
        # pos_idxs the list of index in X which label of values of the index are the same class as anchor
        pos_idxs = np.where(y == y[idx])[0] #y[idx] cooresponding y label
        pos_idx = np.random.choice(pos_idxs[pos_idxs != idx])
        positive.append(X[pos_idx])
        
        # Select a negative sample (sample of a different class from the anchor)
        # neg_idxs the list of index in X which label of values of the index are the different class from anchor
        neg_idxs = np.where(y != y[idx])[0]
        neg_idx = np.random.choice(neg_idxs)
        negative.append(X[neg_idx])
    
    return np.array(anchor), np.array(positive), np.array(negative)

def extract_label_features(model, X_train, y_train):
    """
    Extract feature vectors for one sample of each label from the training set using the provided model.

    Parameters:
    - model: object
      The model used to extract features.
    - X_train: np.ndarray
      Feature matrix of the training set.
    - y_train: np.ndarray
      Labels of the training set.

    Returns:
    - label_features_list: list of np.ndarray
      List of feature vectors, one for each unique label.
    - unique_labels: np.ndarray
      Array of unique labels.
    """
    # Extract one sample for each label
    unique_labels = np.unique(y_train)
    # label_samples dictionary give one sample from X_train for each unique label in y_train
    label_samples = {label: X_train[np.where(y_train == label)[0][0]] for label in unique_labels}
    # For each one sample in X_train then feeded to model to get label_features for each class
    label_features = {label: model.forward(sample) for label, sample in label_samples.items()}

    # Store the features in a list
    label_features_list = [label_features[label] for label in unique_labels]

    return label_features_list, unique_labels

def cosine_similarity(v1, v2):
    """
    Compute the cosine similarity between two vectors.
    Cosine similarity tell how similar or different things are

    Parameters:
    - v1: np.ndarray
      The first vector.
    - v2: np.ndarray
      The second vector.

    Returns:
    - similarity: float
      The cosine similarity score between the two vectors.
    """
    dot_product = np.dot(v1, v2.T)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    similarity = dot_product / (norm_v1 * norm_v2)
    return similarity

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