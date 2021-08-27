# -*- coding: utf-8 -*-
"""
SVM Test

@author: david
"""

# 0. Import Libraries

import sys
sys.path.insert(0,'..')

from toolbox.datasets import classification_data_loader

from toolbox.preprocessing import hold_out
from toolbox.preprocessing.encoding import class_label_encode
from toolbox.preprocessing.normalization import normalize

from toolbox.evaluation.classification import confusion_matrix, compute_indices

from toolbox.learning.linear.ols import OrdinaryLeastSquare
# from toolbox.learning.neural_networks.supervised.models.elm import ExtremeLearningMachine
# from toolbox.learning.neural_networks.supervised.models.mlp import MultiLayerPerceptron
# from toolbox.learning.svm.LSSVC import LSSVC
# from toolbox.learning.svm.LSSVC_GPU import LSSVC_GPU

# import numpy as np
# from sklearn.model_selection import train_test_split

from svm import SupportVectorClassifier
import evaluation

# 1. Load dataSet

X_raw, Y = classification_data_loader.load_dataset(dataset='iris')

# 2. Get dataset's info (number of samples, attributes, classes)

ds_info = classification_data_loader.extract_info(X_raw, Y)
N = ds_info['n_samples']
p = ds_info['n_inputs']
Nc = ds_info['n_outputs']

# 3. Pre-process labels

Y = class_label_encode(Y, Nc, label_type='bipolar')

# 4. Split Data Between Train and Test 

X_tr_raw, y_tr, X_ts_raw, y_ts = hold_out.random_subsampling(X = X_raw, Y = Y, 
                                                             train_size=0.8, random_state=1)

# 5. Normalize data

X_tr = normalize(X_tr_raw, norm_type='z_score')
X_ts = normalize(X_ts_raw, norm_type='z_score', X_ref = X_tr_raw)

# Verify data pattern

print(X_tr.shape)
print(X_tr[0:5,:])
print(y_tr.shape)
print(y_tr[0:5,:])

Ntr, p = X_tr.shape

y1_tr = y_tr[:,0]
y1_ts = y_ts[:,0]

# 6. Model Build and Label Estimation

svc = SupportVectorClassifier(C = 10, kernel='rbf', gamma=20, d=2)
#svc = SupportVectorClassifier(C = 100, kernel='polynomial', gamma=20, d=6)

svc.fit(X_tr,y1_tr)
y_svc = svc.predict(X_ts)

acc = evaluation.calculate_accuracy2(y1_ts,y_svc)

print(svc.b)
print(svc.Xsv.shape)
print(y_svc.shape)

print(y1_ts)
print(y_svc)

print(acc)












