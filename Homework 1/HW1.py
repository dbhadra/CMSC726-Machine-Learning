#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 22:37:57 2018

@author: deepayanbhadra
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

# Q5(a) Generating data
m,n = 10,4
x1 = np.random.multivariate_normal(np.zeros(n), np.eye(n), size=m).transpose()
z = np.random.normal(loc=0, scale=0.5, size=(1,m))
theta0 = 2
theta_x1 = np.matrix([1,0.5,0.25,0.125])
# x = np.concatenate((np.ones((1,m)),x),axis=0)
y = np.dot(theta_x1,x1) + theta0 + z

# defining the cost function
def compute_cost(x1,y,theta_x1,theta0):
       return np.sum(np.power(y-np.dot(theta_x1,x1)-theta0,2))/2/m

# initializations
theta_x1 = np.random.randn(1,4)
theta0= np.random.randn(1,1)  
alp = 1e-3
num_iterations = 1000

# Gradient Step
def gradient_step(theta_current_0,theta_current_x1, x1, y, alp):
    
    grad_theta1 = np.zeros((1,n))
    grad_theta0 = 0
    
    idx = np.arange(x1.shape[-1])  
    np.random.shuffle(idx)
    x1 = x1[..., idx]  
    y = y[..., idx]
    
    for k in range(0,10): # Batch size of 10
        
        grad_theta1 += x1[:,k]*float(y[:,k]-(np.dot(theta_current_x1,x1[:,k])+theta_current_0))
        grad_theta0 += (y[:,k]-(np.dot(theta_current_x1,x1[:,k])+theta_current_0))
    
    # Update theta
    theta_updated_x1 = theta_current_x1 + alp * grad_theta1
    theta_updated_0 = theta_current_0 + alp * grad_theta0

    
    return theta_updated_x1,theta_updated_0
    
# gradient descent
cost_vec = []
for i in range(num_iterations):

    cost_vec.append(compute_cost(x1, y, theta_x1,theta0))
    theta_x1,theta0 = gradient_step(theta0,theta_x1, x1, y, alp)

theta_est = np.array(np.concatenate((theta0,theta_x1),axis=1))
plt.plot(cost_vec)

# Generate m new i.i.d. test samples from PX,Y . 
# Use estimated parameters to compute the MSE on the test set

m,n = 10,4
x1 = np.random.multivariate_normal(np.zeros(n), np.eye(n), size=m).transpose()
z = np.random.normal(loc=0, scale=0.5, size=(1,m))
theta0 = 2
theta_x1 = np.matrix([1,0.5,0.25,0.125])
# x = np.concatenate((np.ones((1,m)),x),axis=0)
y = np.dot(theta_x1,x1) + theta0 + z
compute_cost(x1, y, theta_est[:,0:4],theta_est[:,4])


# Repeat parts (a)-(c) using m = 10. How do training and test errors change? Why?



