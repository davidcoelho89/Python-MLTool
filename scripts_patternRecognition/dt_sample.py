##### Import Libraries

import numpy as np

import auxiliary_functions as auxFunc

import evaluation

from dt_classifier import DTC

##### Test with limiar funtion

vet_lbl_ord = np.array([1,1,1,1,1,2,2,2,2,2,1,1,2])
vet_val_ord = np.array([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5])

# Test Function
bias_opt, bias_opt_entropy = auxFunc.select_limiar(vet_lbl_ord,vet_val_ord)

# Results
print('Limiar otimo: ' + str(bias_opt))
print('Entropia Média: ' + str(bias_opt_entropy))

# Train-Test Split

Xtr = np.array([[3,2,7],[4,1,2],[5,6,8],[9,1,7],[5,7,4]])
ytr = np.array([1,1,2,2,2])
ytr = np.reshape(ytr,(5,1))

print('Xtr:')
print(Xtr)
print('Ytr:')
print(ytr)

Xts = np.array([[1,4,5],[7,3,2]])
yts = np.array([1,2])
yts = np.reshape(yts,(2,1))

# Build Model

DTCmodel = DTC()
DTCmodel.fit(X_train = Xtr, y_train = ytr)

# Predict outputs

yh = DTCmodel.predict(X_test = Xts)

# calculate accuracy

acc = evaluation.calculate_accuracy(yts,yh)
print('Accuracy: ' + str(acc))

# Verify Nodes

node = DTCmodel.node_list[0]
print('Verify Node:')
print('Node Depth: ' + str(node.depth))
print('Node Data Indexes: ' + str(node.data_indexes))
print('Node sons: ' + str(node.sonList))
print('Node attribute: ' + str(node.attribute))
print('Node threshold: ' + str(node.threshold))
print('Node is leaf: ' + str(node.is_leaf))
print('Node Label: ' + str(node.label))

###################################################