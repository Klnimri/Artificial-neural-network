#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[3]:


def purlin(n):
    return n


# In[4]:


def linear_associator(W,p):
    a = purelin(np.dot(W, p))
    return a


# In[22]:


def hebb_rule(T, p):
    W = T * np.transpose(p)
    return W


p = np.array([[0.5,0.5],
              [-0.5,-0.5]])
T = np.array([[1],
              [1]])

W = hebb_rule(T, p)
print("Updated weight Matrix W:")
print(W)


# In[28]:


def check_orthogonal(a, b):
    # Compute the matrix multiplication c = a * b^T
    c = np.outer(a,b)

    if np.allclose(c, 0): #Comparison 
        return True #Orth
    else:
        return False #Not orth


a = np.array([[1],
              [-1],
              [-1]])

b = np.array([[1],
              [1],
              [-1]])

result = check_orthogonal(a, b)

if result:
    print("Vectors are orthogonal.")
else:
    print("Vectors are not orthogonal.")


# In[33]:


def check_unitlength(a):
    c = np.outer(a, a)

    if np.allclose(c, 1):
        return True #unit length
    else:
        return False #not of unit length


a = np.array([[0.5],
              [-0.5],
              [0.5],
              [-0.5]])

result = check_unitlength(a)

if result:
    print("Vector is of unit length.")
else:
    print("Vector is not of unit length.")


# In[34]:


def normalizevec(a):
    norm = np.linalg.norm(a)

    if norm != 0:
        normalized_a = a / norm
        return normalized_a
    else:
        return a 

a = np.array([[1],[-1],[-1]]) 

normalized_a = normalizevec(a)
print("Normalized Vector:")
print(normalized_a)


# In[35]:


result = check_unitlength(a)

if result:
    print("Vector is of unit length.")
else:
    print("Vector is not of unit length.")


# In[38]:


p1 = np.array([[1],[-1],[-1]]) # orange
t1 = np.array([[-1]]) # orange
p2 = np.array([[1],[1],[-1]]) # apple 
t2 = np.array([[1]]) # apple 
P = np.c_[p1,p2]
T = np.c_[t1,t2]


# In[39]:


check_orthogonal(p1,p2)


# In[41]:


a1 = check_unitlength(P)


# In[45]:


a2 = normalizevec(p)
print(a2)


# In[46]:


hebb_rule(T,a2)


# In[ ]:


#Student name: Khalid Nimri
#Student ID: 2140145

