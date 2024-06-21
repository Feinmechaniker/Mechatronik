# -*- coding: utf-8 -*-
"""
Created on 01.06.2023

Program history
02.10.2023    V. 00.01    Start

@author: Prof. Grabow (grabow@amesys.de)
"""
import numpy as np

__version__ = '00.01'
__author__ = 'Joe Grabow'

N = 10000
a = 0
b = 0

# RLS Algorithm parameters
lambda_ = 1  # Forgetting factor
delta = 1e-8    # Small positive constant to initialize P

# Initialization
n_params = 2  # Number of parameters (a and b)
P = np.eye(n_params) / delta
theta = np.zeros(n_params)  # Initial guess for [a, b]

# Simulated data (example data)
m1 = np.random.normal(6, 0.05, N)  # Motor voltage
m2 = np.random.normal(0.15, 0.01, N)  # Motor current
m3 = np.random.normal(930, 5, N)  # Angular speed

# Stack m2 and m3 as feature matrix X
X = np.vstack((m2, m3)).T

# RLS algorithm loop
for i in range(len(m1)):
    x = X[i, :].reshape(-1, 1)  # Current feature vector
    y = m1[i]  # Current target value

    # Prediction error
    y_pred = np.dot(theta, x)
    e = y - y_pred

    # Kalman gain vector
    k = P @ x / (lambda_ + x.T @ P @ x)

    # Parameter update
    theta = theta + k.flatten() * e

    # Covariance matrix update
    P = (P - k @ x.T @ P) / lambda_

# Output the estimated parameters
a, b = theta
print("Estimated a:", a)
print("Estimated b:", b)
