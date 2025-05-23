## Traditional ML Models (Scratch)

# linear_regression.py
# Linear Regression from Scratch
# Closed form solution of linear regression
# X is the input feature matrix
# y is the target variable

import numpy as np
class Linear_Regression:
    def __init__(self,reg_lambda=0):
        self.weights = None
        self.reg_lambda = reg_lambda

    def fit(self,X,y):
        b = np.ones((X.shape[0],1))
        X_b = np.hstack((b,X))
        I = np.eye(X_b.shape[1])
        I[0,0] = 0
        self.weights = np.linalg.inv(X_b.T@X_b + self.reg_lambda*I)@X_b.T@y
    
    def predict(self,X):
        b = np.ones((X.shape[0],1))
        X_b = np.hstack((b,X))
        y_pred = X_b@self.weights
        return y_pred
    
    def R2(self,y_true,y_pred):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (ss_res/ss_tot)
    
    def mse(self,y_true,y_pred):
        return np.mean((y_true - y_pred)**2)
    
    def R2_adjusted(self,y_true,y_pred):
        n = len(y_true)
        k = len(self.weights) - 1
        r2 = self.R2(y_true,y_pred)
        return 1 - (1 - r2)*(n-1)/(n-k-1)

def main():
    # Example usage
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    model = Linear_Regression()
    model.fit(X, y)
    predictions = model.predict(X)
    print("Predictions:", predictions)
    print("R^2:", model.R2(y, predictions))
    print("MSE:", model.mse(y, predictions))
    print("Adjusted R^2:", model.R2_adjusted(y, predictions))

if __name__ == "__main__":
    main()