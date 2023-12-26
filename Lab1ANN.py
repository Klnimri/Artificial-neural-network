#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# In[20]:


#Q1
x = np.linspace(-5, 5, 100)

logsigmoid_values = 1 / (1 + np.exp(-x))

plt.figure(figsize=(8, 6))
plt.plot(x, logsigmoid_values, label='logsigmoid(x)')
plt.xlabel('x')
plt.ylabel('logsigmoid(x)')
plt.title('Plot of logsigmoid(x) for x in [-5, 5]')
plt.grid(True)
plt.legend()
plt.show()


# In[6]:


#Q2
n = np.linspace(-5, 5, 100)

hardlim_values = np.where(n <= 0, 1, 0)

plt.figure(figsize=(8, 6))
plt.plot(n, hardlim_values, label='hardlim(n)')
plt.xlabel('n')
plt.ylabel('hardlim(n)')
plt.title('Plot of hardlim(n) for n in [-5, 5]')
plt.grid(True)
plt.legend()
plt.show()


# In[4]:


#Q2
n = np.linspace(-5, 5, 100)

hardlim_values = np.where(n >= 0, 1, 0)

plt.figure(figsize=(8, 6))
plt.plot(n, hardlim_values, label='hardlim(n)')
plt.xlabel('n')
plt.ylabel('hardlim(n)')
plt.title('Plot of hardlim(n) for n in [-5, 5]')
plt.grid(True)
plt.legend()
plt.show()


# In[22]:


#Q3
n = np.linspace(-5, 5, 100)

linear_values = n

plt.figure(figsize=(8, 6))
plt.plot(n, linear_values, label='linear(n)')
plt.xlabel('n')
plt.ylabel('linear(n)')
plt.title('Plot of linear(n) for n in [-5, 5]')
plt.grid(True)
plt.legend()
plt.show()


# In[23]:


#Q4
n = np.linspace(-5, 5, 100)

tanh_values = (np.exp(n) - np.exp(-n)) / (np.exp(n) + np.exp(-n))

plt.figure(figsize=(8, 6))
plt.plot(n, tanh_values, label='tanh(n)')
plt.xlabel('n')
plt.ylabel('tanh(n)')
plt.title('Plot of tanh(n) for n in [-5, 5]')
plt.grid(True)
plt.legend()
plt.show()


# In[24]:


#Q5
n = np.linspace(-5, 5, 100)

positive_linear_values = np.maximum(0, n)

plt.figure(figsize=(8, 6))
plt.plot(n, positive_linear_values, label='positive_linear(n)')
plt.xlabel('n')
plt.ylabel('positive_linear(n)')
plt.title('Plot of positive_linear(n) for n in [-5, 5]')
plt.grid(True)
plt.legend()
plt.show()


# In[ ]:


#Student name: Khalid Nimri
#Student ID: 2140145
#Section: TR3

