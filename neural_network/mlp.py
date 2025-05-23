# mlp_scratch.py
import numpy as np
class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.lr = lr
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU activation
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2
    
    def backward(self, X, y):
        m = y.shape[0]
        dz2 = self.forward(X) - y
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        dz1 = (dz2 @ self.W2.T) * (self.z1 > 0)  # ReLU derivative
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2