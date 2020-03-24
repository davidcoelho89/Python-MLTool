# -*- coding: utf-8 -*-
"""
OLS Module

--- OLS classifier training ---

    [PARout] = ols_train(DATA,PAR)

    Input:
        DATA.
            input = input matrix                        [p x N]
            output = output matrix                      [Nc x N]
        PAR.
            aprox = type of approximation               [cte]
                1 -> W = Y*pinv(X);
                2 -> W = Y*X'/(X*X');
                3 -> W = Y/X;
    Output:
        PARout.
            W = Regression / Classification Matrix      [Nc x p+1]
            
--- OLS classifier test ---

    [OUT] = ols_classify(DATA,PAR)
        Input:
            DATA.
                input = attributes matrix  	            [p x N]
            PAR.
               W = transformation matrix  	            [Nc x p+1]
        Output:
            OUT.
                y_h = classifier's output               [Nc x N]
"""

import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv

def train(DATA,PAR):
    
    Min = DATA['input']
    Mout = DATA['output']
    
    aprox = PAR['aprox']
    
    Min = np.insert(Min,0,1,axis=0)
    
    if (aprox == 1):
        W = np.dot(Mout,pinv(Min))
    elif (aprox == 2):
        mat_inv = np.dot(Min,Min.T)
        mat_inv = inv(mat_inv)
        W = np.dot(Mout,Min.T)
        W = np.dot(W,mat_inv)
    
    PAR['W'] = W
    return PAR

def classify(DATA,PAR):
    
    Min = DATA['input']
    Min = np.insert(Min,0,1,axis=0)
    
    W = PAR['W']
    
    yh = np.dot(W,Min)
    
    OUT = {'y_h':yh}
    
    return OUT