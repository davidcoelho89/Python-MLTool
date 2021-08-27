# Import Libraries

import sys
sys.path.insert(0,'../..')

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from toolbox.datasets.buid_regressors_matrix import arx_regression_matrix_from_miso

from toolbox.learning.linear.ols import OrdinaryLeastSquare

from toolbox.datasets.sysId_data_loader import DcMotor

# #################################

# Load Dataset

signals_path = os.path.realpath('datasets/signals_motor.csv')
signals = pd.read_csv(signals_path,header=None)

Msignals = np.array(signals)

Cl1 = Msignals[:,0]
Ia1 = Msignals[:,1]
Va1 = Msignals[:,2]
Vel1 = Msignals[:,3]

N = len(Vel1)

print("Number of Samples: " + str(N))
print("Load Shape :" + str(Cl1.shape))
print("Current Shape: " + str(Ia1.shape))
print("Voltage Shape: " + str(Va1.shape))
print("Velocity Shape: " + str(Vel1.shape))

fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(9, 7))
axs[0].plot(Va1)
axs[1].plot(Cl1)
axs[2].plot(Ia1)
axs[3].plot(Vel1)

# #################################

# Build Model for Velocity

y_ts = Vel1

lag_y = 2

u_ts = np.zeros((N,2))
u_ts[:,0] = Va1
u_ts[:,1] = Cl1

lag_u = (2,2)

X,y = arx_regression_matrix_from_miso(u_ts,y_ts,lag_y,lag_u)

print(X.shape)
print(y.shape)

# Build Model

ols = OrdinaryLeastSquare()
ols.fit(X, y, in_row=True, bias=False)
w1 = ols.W
print(ols.W)

# Predict Output

y_ols = ols.predict(X, in_row=True, bias=False)

# Validate Prediction

y_val = y_ts[2:]

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(9, 7))
axs[0].plot(y_val)
axs[1].plot(y_ols)

# Calculate RMSE

N2 = len(y_ols)

err_sum = 0

for i in range(N2):
    err_sum = err_sum + (y_val[i] - y_ols[i])**2

MSE = (1/N2)*err_sum
RMSE = MSE**0.5

print('Erros Velocidade: ')

print(RMSE)
print(MSE)

# ##################################################

# Build model for Current

y_ts = Ia1

lag_y = 2

u_ts = np.zeros((N,2))
u_ts[:,0] = Va1
u_ts[:,1] = Cl1

lag_u = (2,2)

X,y = arx_regression_matrix_from_miso(u_ts,y_ts,lag_y,lag_u)

print(X.shape)
print(y.shape)

# Build Model

ols = OrdinaryLeastSquare()
ols.fit(X, y, in_row=True, bias=False)
w2 = ols.W
print(ols.W)

# Predict Output

y_ols = ols.predict(X, in_row=True, bias=False)

# Validate Prediction

y_val = y_ts[2:]

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(9, 7))
axs[0].plot(y_val)
axs[1].plot(y_ols)

# Calculate RMSE

N3 = len(y_ols)

err_sum = 0

for i in range(N3):
    err_sum = err_sum + (y_val[i] - y_ols[i])**2

MSE = (1/N3)*err_sum
RMSE = MSE**0.5

print('Erros Corrente: ')

print(RMSE)
print(MSE)

# ##################################################

# Use same input and compare outputs

# Time Vector
t = np.arange(0.0, 1.0, 0.001)
N = len(t)

# Input Vector

u = np.zeros((N,2))

for i in range(N):
    if i > 50:
        u[i,0] = 2
    if i > 500:
        u[i,1] = 0.5

# Return tuple with two signals: Output Voltage and Current
dcm = DcMotor(t,u,state_0=[0.0,0.0])
Vel2 = dcm.states[0]
Ia2 = dcm.states[1]

# See Output Voltage and Current
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(9, 7))
axs[0].plot(u)
axs[1].plot(Vel2)
axs[2].plot(Ia2)

Vel_pred = np.zeros((N,1))
Ia_pred = np.zeros((N,1))

for i in range(1,N):
    Vel_pred[i] = w1[0]*Vel_pred[i-2] + w1[1]*Vel_pred[i-1] + w1[2]*u[i-2,0] +  w1[3]*u[i-1,0] + w1[4]*u[i-2,1] +  w1[5]*u[i-1,1]
    Ia_pred[i] = w2[0]*Ia_pred[i-2] + w2[1]*Ia_pred[i-1] + w2[2]*u[i-2,0] +  w2[3]*u[i-1,0] + w2[4]*u[i-2,1] +  w2[5]*u[i-1,1]

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(9, 7))
axs[0].plot(u)
axs[1].plot(Vel_pred)
axs[2].plot(Ia_pred)

# ##################################################