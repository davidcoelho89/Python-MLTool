# -*- coding: utf-8 -*-
"""
Statistics Class Module
"""

import numpy as np

class STATSCLASS():
    'Class that holds the statistics of a classifier'
    def __init__(self):
    
        # Inputs
        self.yh = None
        self.y = None
        
        # Statistics of Last turn
        self.confusion_matrix = None
        self.accuracy = None
		
    def calculate(self,y,yh):
			
        # Hold real and estimated labels
        self.y = y
        self.yh = yh
			
		# Get number of samples and classes
        N = y.shape[0]
        Nc = y.shape[1]
			
		# Calculate Confusion Matrix
        Mconf = np.zeros((Nc,Nc))
        for i in range(N):
            j = np.argmax(y[i,:])
            k = np.argmax(yh[i,:])
            Mconf[j,k] = Mconf[j,k] + 1
			
		# Calculate Accuracy
        acc = np.trace(Mconf) / np.sum(Mconf)
			
		# Hold Results
        self.confusion_matrix = Mconf
        self.accuracy = acc