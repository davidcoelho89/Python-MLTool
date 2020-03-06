# -*- coding: utf-8 -*-
"""
Data Functions Module

"""

import numpy as np           # Work with matrices (arrays)
import pandas as pd          # Load csv files
import math                  # Math Operations

def data_class_loading(OPT):
    
    choice = OPT['prob1']
    
    keyList = ["input","output"] 
    dataSet = {key: None for key in keyList}
    
    if (choice == 1): # iris
        df = pd.read_csv("iris.data.csv") # load data in a dataframe
        dataSet['input'] = np.array(df.drop('classe',1)).T
        x = pd.get_dummies(df['classe'])
        output = np.array(x.values)
        output = output.astype('int32')
        N = output.shape[0]
        Nc = output.shape[1]
        for i in range(N):
            for j in range (Nc):
                if (output[i,j] != 1):
                    output[i,j] = -1
        dataSet['output'] = output.T
    
    return dataSet

def normalize(DATA,OPT):
    
    # Initializations
    
    option = OPT['norm']
    data_in = DATA['input'].T
    
    N = data_in.shape[0]
    p = data_in.shape[1]
    
    Xmin = data_in.min(0)
    Xmax = data_in.max(0)
    Xmed = data_in.mean(0)
    dp = data_in.std(0)
    
    # Algorithm
    
    data_norm = np.zeros((N,p))
    
    if(option == 0):   # Don`t apply normalization
        data_norm = data_in
    elif(option == 1): # ToDo: normalize between [0 1]
        data_norm = data_in
    elif(option == 2): # ToDo: normalize between [-1 1]
        data_norm = data_in
    elif(option == 3): # normalize by z-score transform (mean and st)
        data_norm = (data_in - Xmed)/dp
    
    data_norm = data_norm.T
    
    # Output Sctructure
    
    DATA['input'] = data_norm
    DATA['Xmin'] = Xmin
    DATA['Xmax'] = Xmax
    DATA['Xmed'] = Xmed
    DATA['dp'] = dp
    
    return DATA
    
def label_adjust(DATA,OPT):
    
    choice = OPT['lbl']
    labels_in = DATA['output']
    
    if(choice == 0):
        labels_out = labels_in
        
    DATA['output'] = labels_out
    
    return DATA

def hold_out(DATA,OPT):
    
    hold = OPT['hold']
    ptrn = OPT['ptrn']
    
    Min = DATA['input']
    Mout = DATA['output']
    
    N = Min.shape[1]
    
    if(hold == 1):
        
        # Shuffle data
        I = np.random.permutation(N)
        Min = Min[:,I]
        Mout = Mout[:,I]
        
        # Number of samples for training
        Ntr = math.floor(ptrn*N)
        
        # Samples for training and test
        Min_tr = Min[:,0:Ntr]
        Mout_tr = Mout[:,0:Ntr]
        Min_ts = Min[:,Ntr:N]
        Mout_ts = Mout[:,Ntr:N]
        
    DATAtr = {'input': Min_tr,'output':Mout_tr}
    DATAts = {'input': Min_ts,'output':Mout_ts}
    
    DATAout = {'DATAtr':DATAtr,'DATAts':DATAts}
    
    return DATAout
