# -*- coding: utf-8 -*-
"""
OLS Classifier Class Module
"""

import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv

class OLS:
    'Class that implements the Ordinary Least Squares Classifier'
    
    def __init__(self, aprox=1):
        
        # Model Hyperparameters
        self.aprox = aprox
        
        # Data used for model building
        self.x = None
        self.y = None
        
        # Model Parameters
        self.W = None
        
    def fit(self,X,Y,verboses=0):
	    # Hold data used for model building
        self.x = X
        self.y = Y
		
		# Use [p+1 x N] and [Nc x N] notation
        X = X.T
        X = np.insert(X,0,1,axis=0)
        Y = Y.T
		
		# Calculate W [Nc x p+1]
        if (self.aprox == 1):
            self.W = np.dot(Y,pinv(X))
        elif (self.aprox == 2):
            Minv = np.dot(X,X.T)
            Minv = inv(Minv)
            self.W = np.dot(Y,X.T)
            self.W = np.dot(self.W,Minv)
        
    def predict(self,X):
		
		# Use [p x N] notation
        X = X.T
        X = np.insert(X,0,1,axis=0)
		
		# Algorithm
        yh = np.dot(self.W,X)
		
		# Use [N x Nc] notation
        yh = yh.T
		
        return yh        