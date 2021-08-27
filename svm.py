# -*- coding: utf-8 -*-
"""
SVM Test

@author: david
"""

import numpy as np
import math
from numpy.linalg import norm

from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

class SupportVectorClassifier:
    """Class that implements the Support Vector Classifier"""

    def __init__(self, C = 10, kernel='rbf', gamma=2, d=2, l=1e-4):
        # Data points organized as line-vectors
        self.X = None   # Input       [n x p]
        self.Y = None   # Output      [n x 1] [-1 or 1]
        
        # Hyperparameters
        self.C = C
        self.kernel_type = kernel
        self.gamma = gamma
        self.d = d
        self.l = l

        # Parameters
        self.K = None        # Kernel Matrix
        self.alphas = None   # lagrange multiplies
        self.Xsv = None      # Inputs of support vectors
        self.Ysv = None      # Outputs of support vectors
        self.b = None        # Bias of SVM

    def fit(self, X, Y):
		
        # Keep samples
        self.X = X
        self.Y = Y

        N, _ = X.shape
        
        # Calculate Kernel Matrix
        self.K = self.svc_kernel_mat(self)
        
        # Adjust Outputs for cvxopt function
        A = Y.reshape(-1,1) * 1
        A = A.reshape(1,-1)
        A = A.astype('float')
        
        # Calculate Optmization Matrices
        P = cvxopt_matrix(self.K)
        q = cvxopt_matrix(-np.ones((N, 1)))
        G = cvxopt_matrix(np.vstack((np.eye(N)*-1,np.eye(N))))
        h = cvxopt_matrix(np.hstack((np.zeros(N), np.ones(N) * self.C)))
        A = cvxopt_matrix(A)
        b = cvxopt_matrix(np.zeros(1))
        
        #Run solver
        cvxopt_solvers.options['show_progress'] = False
        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])
        
        # Get support vectors (alphas > epsilon)
        epsilon = 1e-4
        sv = (alphas > epsilon).flatten()

        Xsv = X[sv,:]
        Ysv = Y[sv]
        alphas = alphas[sv,0]
        Nsv = len(Ysv)
        
        # Calculate Bias
        b = 0
        for i in range(Nsv):
            b = b + 1/Ysv[i]
            for j in range(Nsv):
                kij = self.kernel_func(Xsv[j,:],Xsv[i,:],
                                       kernel = self.kernel_type,
                                       gamma = self.gamma,
                                       d = self.d)
                b = b - Ysv[j]*alphas[j]*kij

        # Hold Results
        self.Xsv = Xsv
        self.Ysv = Ysv
        self.alphas = alphas
        self.b = b

    def predict(self, X):
        
        N, _ = X.shape
        
        Xsv = self.Xsv
        Ysv = self.Ysv
        alphas = self.alphas
        b = self.b
        Nsv = len(Ysv)
        
        Y_pred = np.zeros(N)
        
        for i in range(N):
            xts = X[i,:]
            Kvet = np.zeros(Nsv)
            for j in range(Nsv):
                kij = self.kernel_func(xts,Xsv[j,:],
                                       kernel = self.kernel_type,
                                       gamma = self.gamma,
                                       d = self.d)
                Kvet[j] = alphas[j]*Ysv[j]*kij

            Wx = np.sum(Kvet)
            Y_pred[i] = Wx + b
        
        return Y_pred

        
    @staticmethod
    def kernel_func(xi,xj,kernel='rbf',gamma=2,d=2):
        if (kernel == 'linear'):
            return np.dot(xi,xj)
        elif (kernel == 'polynomial'):
            return (1 + np.dot(xi,xj)) ** d
        elif (kernel == 'rbf'):
            return math.exp(-(norm(xi-xj)**2)/(gamma**2))
    
    @staticmethod
    def svc_kernel_mat(self):
        
        X = self.X
        Y = self.Y
        kernel_type = self.kernel_type
        gamma = self.gamma
        d = self.d
        l = self.l
        
        N, _ = X.shape
        
        K = np.zeros((N,N))
        
        for i in range(N):
            xi = X[i,:]
            yi = Y[i]
            for j in range(i,N,1):
                xj = X[j,:]
                yj = Y[j]
                K[i,j] = yi*yj*self.kernel_func(xi,xj,kernel=kernel_type,
                                                gamma=gamma,d=d)
                K[j,i] = K[i,j]
        
        K = K + l*np.eye(N,N)
        
        return K

        













