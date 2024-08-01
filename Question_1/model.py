import numpy as np

class SVM:
    """
    A class that implements a Support Vector Machine (SVM) for binary classification.

    Attributes:
    -----------
    lr : float
        Learning rate for gradient descent.
    lambda_param : float
        Regularization parameter to prevent overfitting.
    n_iters : int
        Number of iterations for training.
    w : np.ndarray or None
        Weight vector of the SVM model.
    b : float or None
        Bias term of the SVM model.

    Methods:
    --------
    fit(X, y):
        Trains the SVM model on the given data.
    predict(X):
        Predicts class labels for the given data.
    """

    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=10):
        """
        Initializes the SVM with specified hyperparameters.

        Parameters:
        -----------
        learning_rate : float, optional
            The step size for gradient descent optimization (default is 0.01).
        lambda_param : float, optional
            Regularization parameter (default is 0.01).
        n_iters : int, optional
            Number of iterations for training (default is 10).
        """
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Trains the SVM model using the provided training data.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Label vector of shape (n_samples,) with binary labels (-1 or 1).

        Updates:
        --------
        self.w : np.ndarray
            Weight vector learned during training.
        self.b : float
            Bias term learned during training.
        """
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Training with gradient descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # If correctly classified, update weights for regularization
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # If misclassified, update weights and bias for hinge loss(penalty for misclassification) and the regularization term)
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        """
        Predicts the class labels for the given data.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns:
        --------
        np.ndarray
            Predicted class labels of shape (n_samples,). 
            Values are -1 or 1 based on the sign of the decision function.
        """
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)

class MulticlassSVM:
    """
    A class that implements a multiclass SVM using the One-vs-All strategy.

    Attributes:
    -----------
    learning_rate : float
        Learning rate for gradient descent.
    lambda_param : float
        Regularization parameter to prevent overfitting.
    n_iters : int
        Number of iterations for training.
    models : list of SVM
        List of binary SVM models, one for each class.

    Methods:
    --------
    fit(X, y):
        Trains a set of binary SVM models for each class.
    predict(X):
        Predicts class labels for the given data using the trained models.
    """

    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=10):
        """
        Initializes the MulticlassSVM with specified hyperparameters.

        Parameters:
        -----------
        learning_rate : float, optional
            The step size for gradient descent optimization (default is 0.01).
        lambda_param : float, optional
            Regularization parameter (default is 0.01).
        n_iters : int, optional
            Number of iterations for training (default is 10).
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.models = []

    def fit(self, X, y):
        """
        Trains a binary SVM model for each class using the One-vs-All approach.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Label vector of shape (n_samples,) with multiclass labels.

        Updates:
        --------
        self.models : list of SVM
            List of trained binary SVM models, one for each class.
        """
        self.classes = np.unique(y)
        for c in self.classes:
            y_binary = np.where(y == c, 1, -1)
            model = SVM(self.learning_rate, self.lambda_param, self.n_iters)
            model.fit(X, y_binary)
            self.models.append(model)

    def predict(self, X):
        """
        Predicts the class labels for the given data using the trained binary SVM models.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns:
        --------
        np.ndarray
            Predicted class labels of shape (n_samples,). Each label corresponds to the class with the highest score from the binary classifiers.
        """
        predictions = np.array([model.predict(X) for model in self.models])
        return self.classes[np.argmax(predictions, axis=0)]
