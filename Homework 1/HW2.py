#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 20:40:30 2018

@author: deepayanbhadra
"""

# imports
import numpy as np
import scipy as sc
import numpy.linalg as LA
from sklearn.utils import shuffle

# Generating Data: Two Classes are generated from two different Gaussian Clusters 

d = 5  # dimension 
m = 4096*2 # number of training samples 
c = 0.5 # parameter controlling seperation between the clusters
mu1 = c*np.array([1,1,1,1,1])
mu2 = -mu1 
sigma_squared_1 = 1
sigma_squared_2 = sigma_squared_1 

X_train_1 = np.random.multivariate_normal(mu1,sigma_squared_1*np.identity(d),int(m/2)) 
X_train_2 = np.random.multivariate_normal(mu2,sigma_squared_2*np.identity(d),int(m/2)) 
X_train = np.concatenate((X_train_1,X_train_2),axis=0)

Y_train_1 = np.ones((int(m/2),1))
Y_train_2 = 0*np.ones((int(m/2),1))
Y_train = np.concatenate((Y_train_1,Y_train_2),axis=0)

X_train = np.column_stack((np.ones((X_train.shape[0],1)),X_train)) # adding 1 as X0

plt.scatter(X_train_1[:,0],X_train_1[:,1], color='b',marker='o')
plt.scatter(X_train_2[:,0],X_train_2[:,1], color='r',marker='o')
plt.title('Two Gaussian clusters with different labels')
plt.xlabel('x1')
plt.ylabel('x2')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cross_entropy(y,y_est):
    return -y*np.log(y_est)-(1-y)*np.log(1-y_est)

def compute_cost(X,Y,theta):
    m = X.shape[0] 
    loss= 0 
    for i in range(m):
        loss += cross_entropy(Y[i],sigmoid(np.dot(X[i,:],theta)))  
    return loss 

# Stochastic Gradient Descent 

def gradient_step_log_reg(theta_current, X, Y, learning_rate):
    
    grad = np.zeros_like(theta_current)
    
    idx = np.arange(X.shape[0])  
    np.random.shuffle(idx)
    X = X[idx,...]  
    Y = Y[idx,...]
    
    for i in range(0,32): # Batch size
        grad += (sigmoid(np.dot(theta_current,X[i,:]))-Y[i])*X[i,:]
        
        
    theta_current += -learning_rate*grad 
    return theta_current

# initializations
theta = np.random.randn(6)
learning_rate = 1e-5
num_iterations = 1000

# shuffle the data set 
X_train, Y_train = shuffle(X_train, Y_train)

# gradient descent
cost_vec = []
for t in range(num_iterations):
    theta = gradient_step_log_reg(theta, X_train, Y_train, learning_rate)
    if t%25==0:
        cost_vec.append(compute_cost(X_train, Y_train, theta)) 
        
iterations = np.linspace(0,num_iterations, num=len(cost_vec))
plt.plot(iterations,cost_vec)
plt.xlabel('iterations')
plt.ylabel('cost')


# Visualize the final seperating line

plt.scatter(X_train_1[:,0],X_train_1[:,1], color='b',marker='o')
plt.scatter(X_train_2[:,0],X_train_2[:,1], color='r',marker='o')
plt.title('Logistic regression line seperating the two training classes')
plt.plot([-4,4], (np.array([-4,4])*-theta[1]-theta[0])/theta[2],'r')
plt.xlabel('x1')
plt.ylabel('x2')


# Test Dataset

X_test_1 = np.random.multivariate_normal(mu1,sigma_squared_1*np.identity(d),int(m/2)) 
X_test_2 = np.random.multivariate_normal(mu2,sigma_squared_2*np.identity(d),int(m/2)) 
X_test = np.concatenate((X_test_1,X_test_2),axis=0)

Y_test_1 = np.ones((int(m/2),1))
Y_test_2 = 0*np.ones((int(m/2),1))
Y_test = np.concatenate((Y_test_1,Y_test_2),axis=0)

X_test = np.column_stack((np.ones((X_test.shape[0],1)),X_test)) # adding 1 as X0

plt.scatter(X_test_1[:,0],X_test_1[:,1], color='b',marker='o')
plt.scatter(X_test_2[:,0],X_test_2[:,1], color='r',marker='o')
plt.title('Two Gaussian test clusters with different labels')
plt.xlabel('x1')
plt.ylabel('x2')

# Loss on the test set

print(compute_cost(X_test, Y_test, theta))

# Visualize the final seperating line on test data

plt.scatter(X_test_1[:,0],X_test_1[:,1], color='b',marker='o')
plt.scatter(X_test_2[:,0],X_test_2[:,1], color='r',marker='o')
plt.title('Logistic regression line seperating the two test classes')
plt.plot([-4,4], (np.array([-4,4])*-theta[1]-theta[0])/theta[2],'r')
plt.xlabel('x1')
plt.ylabel('x2')




