# -*- coding: utf-8 -*-
"""
Class Statistics Module

"""

import numpy as np           # Work with matrices (arrays)

def stats_1turn(DATA,OUT):
    
    y_h = OUT['y_h']
    Mout = DATA['output']
    
    Nc = Mout.shape[0]
    N = Mout.shape[1]
    
    Mconf = np.zeros((Nc,Nc))
    
    for i in range(N):
        j = np.argmax(Mout[:,i])
        k = np.argmax(y_h[:,i])
        Mconf[j,k] = Mconf[j,k] + 1
    
    acc = np.trace(Mconf) / np.sum(Mconf)
    
    STATS = {'Mconf': Mconf, 'acc': acc}
    
    return STATS   

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