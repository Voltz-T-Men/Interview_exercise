import config
import numpy as np
from utils import *

class NeuralNetwork:
    """
    A simple neural network with one hidden layer for training using triplet loss.

    Attributes:
    -----------
    input_size : int
        The size of the input layer.
    hidden_size : int
        The size of the hidden layer.
    output_size : int
        The size of the output layer.
    learning_rate : float
        The learning rate for weight updates.

    Methods:
    --------
    relu(z):
        Apply the ReLU activation function.
    forward(X):
        Perform the forward pass.
    compute_loss(anchor, positive, negative, margin=1.0):
        Compute the triplet loss.
    backward(anchor, positive, negative, alpha=1.0, learning_rate=0.01):
        Perform the backward pass and update weights using gradient descent.
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        """
        Initialize the neural network with the given layer sizes and learning rate.

        Parameters:
        -----------
        input_size : int
            The size of the input layer.
        hidden_size : int
            The size of the hidden layer.
        output_size : int
            The size of the output layer.
        learning_rate : float, optional
            The learning rate for weight updates (default is 0.001).
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def relu(self, z):
        """
        Apply the ReLU activation function.

        Parameters:
        -----------
        z : numpy array
            Input to the activation function.

        Returns:
        --------
        numpy array
            Output with ReLU applied.
        """
        return np.maximum(0, z)
    
    def forward(self, X):
        """
        Perform the forward pass.

        Parameters:
        -----------
        X : numpy array
            Input data.

        Returns:
        --------
        numpy array
            Output of the forward pass.
        """
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.tanh(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.Z2  # Output layer (no activation for triplet loss)
        return self.A2

    def compute_loss(self, anchor, positive, negative, margin=1.0):
        """
        Compute the triplet loss.

        Parameters:
        -----------
        anchor : numpy array
            Anchor samples.
        positive : numpy array
            Positive samples.
        negative : numpy array
            Negative samples.
        margin : float, optional
            Margin for the triplet loss (default is 1.0).

        Returns:
        --------
        float
            Triplet loss value.
        """
        anchor_output = self.forward(anchor)
        positive_output = self.forward(positive)
        negative_output = self.forward(negative)
        loss = triplet_loss(anchor_output, positive_output, negative_output, margin)
        return loss

    def backward(self, anchor, positive, negative, learning_rate=0.01):
        """
        Perform the backward pass and update weights using gradient descent.

        Parameters:
        -----------
        anchor : numpy array
            Anchor samples.
        positive : numpy array
            Positive samples.
        negative : numpy array
            Negative samples.
        alpha : float, optional
            Margin for the triplet loss (default is 1.0).
        learning_rate : float, optional
            Learning rate for weight updates (default is 0.01).
        """
        # Forward pass to get outputs for anchor, positive, and negative samples
        anchor_output = self.forward(anchor)
        positive_output = self.forward(positive)
        negative_output = self.forward(negative)
        
        #-------------Calculate the gradients for the loss with respect to outputs-------------
        pos_dist = 2 * (anchor_output - positive_output)  # Gradient of positive distance
        neg_dist = 2 * (anchor_output - negative_output)  # Gradient of negative distance
        
        dloss_da = pos_dist - neg_dist  # Gradient of anchor output
        dloss_dp = -pos_dist  # Gradient of positive output
        dloss_dn = neg_dist  # Gradient of negative output
        #--------------------------------------------------------------------------------------

        # Update weights and biases for the output layer
        self.W2 -= learning_rate * np.dot(self.A1.T, dloss_da)  # Gradient descent update for W2
        self.b2 -= learning_rate * np.sum(dloss_da, axis=0, keepdims=True)  # Gradient descent update for b2
        
        # Compute gradients for the hidden layer
        dW1_a = np.dot(anchor.T, np.dot(dloss_da, self.W2.T) * (self.Z1 > 0))  # Gradient of W1 for anchor
        db1_a = np.sum(np.dot(dloss_da, self.W2.T) * (self.Z1 > 0), axis=0, keepdims=True)  # Gradient of b1 for anchor
        
        dW1_p = np.dot(positive.T, np.dot(dloss_dp, self.W2.T) * (self.Z1 > 0))  # Gradient of W1 for positive
        db1_p = np.sum(np.dot(dloss_dp, self.W2.T) * (self.Z1 > 0), axis=0, keepdims=True)  # Gradient of b1 for positive
        
        dW1_n = np.dot(negative.T, np.dot(dloss_dn, self.W2.T) * (self.Z1 > 0))  # Gradient of W1 for negative
        db1_n = np.sum(np.dot(dloss_dn, self.W2.T) * (self.Z1 > 0), axis=0, keepdims=True)  # Gradient of b1 for negative
        
        # Update weights and biases for the hidden layer
        self.W1 -= learning_rate * (dW1_a + dW1_p + dW1_n)  # Gradient descent update for W1 using the Gradient of W1 of those triplet
        self.b1 -= learning_rate * (db1_a + db1_p + db1_n)  # Gradient descent update for b1 using the Gradient of b1 of those triplet

def train(X_train, y_train):
    """
    Train the neural network using the training data.

    Parameters:
    -----------
    X_train : np.ndarray
        Training images.
    y_train : np.ndarray
        Training labels.

    Returns:
    --------
    NeuralNetwork
        Trained neural network model.
    """
    model = NeuralNetwork(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)
    for epoch in range(config.EPOCHS):
        epoch_loss = 0
        num_batches = len(X_train) // config.BATCH_SIZE
        for _ in range(num_batches):
            anchor, positive, negative = create_triplets(X_train, y_train, config.BATCH_SIZE)
            loss = model.compute_loss(anchor, positive, negative)
            epoch_loss += np.sum(loss)
            model.backward(anchor, positive, negative, config.LR)
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{config.EPOCHS}, Loss: {avg_loss:.4f}")
    print("Training completed.")
    return model
