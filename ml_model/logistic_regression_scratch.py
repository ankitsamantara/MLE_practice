## Traditional ML Models (Scratch)

# logistic_regression.py
# Import necessary libraries
import numpy as np

# Define the Logistic Regression class
class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        """
        Initialize the Logistic Regression model.

        Parameters:
        lr (float): Learning rate for gradient descent.
        epochs (int): Number of iterations for training.
        """
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        """
        Compute the sigmoid function.

        Parameters:
        z (numpy.ndarray): Input array.

        Returns:
        numpy.ndarray: Sigmoid of the input.
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Train the Logistic Regression model using gradient descent.

        Parameters:
        X (numpy.ndarray): Feature matrix (m x n), where m is the number of samples and n is the number of features.
        y (numpy.ndarray): Target vector (m,).
        """
        # Initialize weights (theta) to zeros
        self.theta = np.zeros(X.shape[1])

        # Perform gradient descent for the specified number of epochs
        for _ in range(self.epochs):
            # Compute predictions using the sigmoid function
            preds = self.sigmoid(X @ self.theta)

            # Compute the gradient of the loss function with respect to weights
            gradient = X.T @ (preds - y) / len(y)

            # Update weights using the gradient and learning rate
            self.theta -= self.lr * gradient

    def predict(self, X):
        """
        Make predictions using the trained Logistic Regression model.

        Parameters:
        X (numpy.ndarray): Feature matrix (m x n).

        Returns:
        numpy.ndarray: Predicted labels (m,).
        """
        # Compute probabilities and convert to binary predictions (0 or 1)
        return (self.sigmoid(X @ self.theta) > 0.5).astype(int)
    
    def predict_proba(self, X):
        """
        Compute the predicted probabilities for each class.

        Parameters:
        X (numpy.ndarray): Feature matrix (m x n).

        Returns:
        numpy.ndarray: Predicted probabilities (m,).
        """
        # Compute probabilities using the sigmoid function
        return self.sigmoid(X @ self.theta)
    
    def accuracy(self, y_true, y_pred):
        """
        Compute the accuracy of the model.

        Parameters:
        y_true (numpy.ndarray): True labels (m,).
        y_pred (numpy.ndarray): Predicted labels (m,).

        Returns:
        float: Accuracy score.
        """
        return np.mean(y_true == y_pred)

    def precision(self, y_true, y_pred):
        """
        Compute the precision of the model.

        Parameters:
        y_true (numpy.ndarray): True labels (m,).
        y_pred (numpy.ndarray): Predicted labels (m,).

        Returns:
        float: Precision score.
        """
        true_positive = np.sum((y_true == 1) & (y_pred == 1))
        predicted_positive = np.sum(y_pred == 1)
        return true_positive / predicted_positive if predicted_positive > 0 else 0
    
    def recall(self, y_true, y_pred):
        """
        Compute the recall of the model.

        Parameters:
        y_true (numpy.ndarray): True labels (m,).
        y_pred (numpy.ndarray): Predicted labels (m,).

        Returns:
        float: Recall score.
        """
        true_positive = np.sum((y_true == 1) & (y_pred == 1))
        actual_positive = np.sum(y_true == 1)
        return true_positive / actual_positive if actual_positive > 0 else 0
    
    def f1_score(self, y_true, y_pred):
        """
        Compute the F1 score of the model.

        Parameters:
        y_true (numpy.ndarray): True labels (m,).
        y_pred (numpy.ndarray): Predicted labels (m,).

        Returns:
        float: F1 score.
        """
        p = self.precision(y_true, y_pred)
        r = self.recall(y_true, y_pred)
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    

# Import metrics for evaluation
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 2)  # 100 samples, 2 features
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Label: 1 if sum of features > 1, else 0

    # Add bias term (intercept)
    X = np.c_[np.ones(X.shape[0]), X]

    # Initialize and train the model
    model = LogisticRegression(lr=0.1, epochs=10)
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)

    # Compute metrics
    accuracy = model.accuracy(y, y_pred)
    precision = model.precision(y, y_pred)
    recall = model.recall(y, y_pred)
    f1 = model.f1_score(y, y_pred)

    # Print results
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

if __name__ == "__main__":
    main()