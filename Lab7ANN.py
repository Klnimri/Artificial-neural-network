#!/usr/bin/env python
# coding: utf-8

# In[67]:


import numpy as np


# In[68]:


#Q1 a:
def logsigmoid(n):
    output = 1 / (1 + np.exp(-n))
    return output

def logsiglayer(W1, p, b1):
    p_reshaped = np.array([p]).reshape(-1, 1)
    a1 = logsigmoid(np.dot(W1, p_reshaped) + b1)
    return a1


# In[69]:


#Q1 b:
def purelin(n):
    return n

def linearlayer(W2, a1, b2):
    a2 = purelin(np.dot(W2, a1) + b2)
    return a2


# In[70]:


#Q2: 
def forwardpropagation(p, W1, b1, W2, b2):
    a0 = np.array([p]).reshape(-1, 1)
    a1 = logsiglayer(W1, a0, b1)
    a2 = linearlayer(W2, a1, b2)
    return a0, a1, a2


# In[71]:


#Q3: 
def dlogsig(n):
    a1 = 1 / (1 + np.exp(-n))
    derivative = (1 - a1) * a1
    return derivative

def dpurelin(n):
    derivative = np.ones_like(n)
    return derivative


# In[72]:


#Q4:
def backpropagation(F2, t, a2, F1, W2):
    s2 = -2 * F2 * (a2 - t)
    s1 = np.dot(F1, W2.T) * s2
    return s2, s1


# In[73]:


#Q5:
def updateparams(W1, W2, b1, b2, s1, s2, a0, a1, a2, alpha):
    W2 = W2 - alpha * np.dot(s2, a1)
    b2 = b2 - alpha * s2
    W1 = W1 - alpha * np.outer(s1.T, a0)
    b1 = b1 - alpha * s1
    return W1, W2, b1, b2


# In[74]:


#Q6:
def backpropagatealgorithm(W1, W2, b1, b2, p, t, alpha):
    a0, a1, a2 = forwardpropagation(p, W1, b1, W2, b2)
    F2 = np.diag(dlogsig(a2))
    F1 = np.diag(dpurelin(a1))
    s2, s1 = backpropagation(F2, t, a2, F1, W2)
    W1, W2, b1, b2 = updateparams(W1, W2, b1, b2, s1, s2, a0, a1, a2, alpha)
    return W1, W2, b1, b2


# In[76]:


#Q7:
np.random.seed(42)
W1 = np.random.rand(2, 1)
b1 = np.random.rand(1, 2)
W2 = np.random.rand(1, 2)
b2 = np.random.rand(1, 1)

data_points = [(-2, 0.0), (-1.5, 0.006), (-1, 0.29), (-0.5, 0.617), (0, 1.0),
               (0.5, 1.38), (1, 1.707), (1.5, 1.92), (2, 2.0)]

max_iterations = 50
alpha = 0.1

for iteration in range(max_iterations):
    idx = np.random.randint(len(data_points))
    p, t = data_points[idx]
    W1, W2, b1, b2 = backpropagatealgorithm(W1, W2, b1, b2, p, t, alpha)

print(f"Converged after {iteration + 1} iterations")
print("Converged values of W1:\n", W1)
print("Converged values of W2:\n", W2)
print("Converged values of b1:\n", b1)
print("Converged values of b2:\n", b2)


# In[ ]:


#Student name: Khalid Nimri
#Student ID: 2140145

