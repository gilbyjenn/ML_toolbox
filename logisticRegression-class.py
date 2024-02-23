### Class to perform gradient-descent based Logistic Regression  for UNCC BINF6210 Machine Learning class.
### Jennifer Gilby, Fall 2023
### Completed as a group project with Whitney Brannen, Lydia Holley, & Jimmy Nguyen

import numpy as np
import pandas as pd

### logistical regression class for classification
class MyLogisticRegression:
    
    def __init__(self, learning_rate=0.01, max_iters=10000, delta_J=0.00001):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.delta_J = delta_J
        self.theta = None
        self.mean = None
        self.std = None
        self.standardized_X = None
    
    # standardize the data
    def standardize(self, X):        
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.standardized_X = (X - self.mean) / self.std
        
        return self.standardized_X
    
    # find sigmoid value
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # find cost
    def cost_function(self, X, y, theta):
        m = len(y)
        h = self.sigmoid(np.dot(X, theta))
        log_term_1 = np.log(h)
        log_term_2 = np.log(1 - h + 1e-10) # add constant to avoid dividing by 0 error

        J = (-1 / m) * (np.dot(y.T, log_term_1) + np.dot((1 - y).T, log_term_2))
        return J

    # perform gradient descent
    def gradient_descent(self, X, y, theta):
        m = len(y)
        h = self.sigmoid(np.dot(X, theta))
        gradient = (1 / m) * (np.dot(X.T, (h - y)))
        return gradient

    def fit(self, X, y):
        #standardize
        X = self.standardize(X)
        
        #initialize values
        X = np.insert(X, 0, 1, axis=1)  # Add a bias term
        m, n = X.shape
        self.theta = np.zeros(n)

        # set previous cost to infinity
        # find the delta J values until stop criteria is met
        prev_cost = float('inf')
        for i in range(self.max_iters):
            gradient = self.gradient_descent(X, y, self.theta)
            self.theta -= self.learning_rate * gradient

            current_cost = self.cost_function(X, y, self.theta)

            if abs(prev_cost - current_cost) < self.delta_J:
                print(f"Converged after {i+1} iterations.")
                break

            prev_cost = current_cost

        if i == self.max_iters - 1:
            print("Warning: Maximum number of iterations reached without convergence.")

    # predict method
    def predict(self, X):
        #standardize
        X = self.standardize(X)
        
        X = np.insert(X, 0, 1, axis=1)  # Add a bias term
        predictions = self.sigmoid(np.dot(X, self.theta))
        return (predictions >= 0.5).astype(int)