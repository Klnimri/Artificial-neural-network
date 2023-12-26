#!/usr/bin/env python
# coding: utf-8

# In[59]:


import numpy as np
import matplotlib.pyplot as plt


# In[60]:


def logsigmoid(n):
    output = 1 / (1 + np.exp(-n))
    return output

def logsiglayer(W1, p, b1):
    a1 = logsigmoid(np.dot(W1, p) + b1)
    
    return a1


# In[61]:


def purelin(n):
    return n

def linearlayer(W2, a1, b2):
    a2 = purelin(np.dot(W2, a1) + b2)

    return a2


# In[88]:


W1 = np.array([[10, 10]])   # Layer 1
b1 = np.array([[-10],
               [10]])

W2 = np.array([1, 1])       # Layer 2
b2 = np.array([[0],[0]])


# In[110]:


p_values = np.linspace(-2, 2, num=2).reshape(2, 1)


# In[111]:


a1 = logsiglayer(W1, p_values, b1)


# In[112]:


a2 = linearlayer(W2,a1,b2)


# In[113]:


plt.figure(figsize=(8, 6))
plt.plot(p_values, p_values, label='p_values', linestyle='--', marker='o', markersize=5)
plt.plot(p_values, a2, label='a2', linestyle='-', marker='s', markersize=5)
plt.xlabel('p_values')
plt.ylabel('Values')
plt.title('p_values vs a2')
plt.legend()
plt.grid(True)
plt.show()


# In[167]:


W1 = np.array([[10, 10]])   # Layer 1
b1 = np.array([[-10],
               [10]])

W2 = np.array([1, 1])       # Layer 2
b2 = np.array([[0],[0]])


# In[168]:


b1_1_values = np.linspace(0, 20, num=1)
p_values = np.linspace(-2, 2, num=2).reshape(-1,1)

for b1_1 in b1_1_values:
    b1 = np.array([[b1_1, 10], [-10, 10]])
    a2_values = []

    for p in p_values:
        a1 = logsiglayer(w1, p_values, b1)
        a2 = linearlayer(w2, a1, b2)

        a2_values.append(a2)

    a2_values = np.array(a2_values).reshape(-1,1)

    plt.figure()
    plt.plot(p_values, label='p_values', linestyle='--', marker='o', markersize=5)
    plt.plot(a2_values, label='a2', linestyle='-', marker='s', markersize=5)
    plt.xlabel('p')
    plt.ylabel('a2')
    plt.title(f'p vs a2 for b1,1 = {b1_1}')
    plt.legend()
    plt.grid(True)
    plt.show()


# In[178]:


W1 = np.array([[10, 10]])   # Layer 1
b1 = np.array([[-10],
               [10]])

W2 = np.array([1, 1])       # Layer 2
b2 = np.array([[0],[0]])

b1_2_values = np.linspace(0, 20, num=1)
p_values = np.linspace(-2, 2, num=2).reshape(-1,1)

for b1_2 in b1_2_values:
    b2 = np.array([[b1_1, 10], [-10, 10]])
    a2_values = []

    for p in p_values:
        a1 = logsiglayer(w1, p_values, b2)
        a2 = linearlayer(w2, a1, b2)

        a2_values.append(a2)

    a2_values = np.array(a2_values).reshape(-1,1)

    plt.figure()
    plt.plot(p_values, label='p_values', linestyle='--', marker='o', markersize=5)
    plt.plot(a2_values, label='a2', linestyle='-', marker='s', markersize=5)
    plt.xlabel('p')
    plt.ylabel('a2')
    plt.title(f'p vs a2 for b1,1 = {b12}')
    plt.legend()
    plt.grid(True)
    plt.show()


# In[180]:


W1 = np.linspace(-1, 1, num=2).reshape(-1, 1)
b1 = np.array([[-10],
               [10]])

W2 = np.array([1, 1])  # Layer 2
b2 = np.array([[0], [0]])

p_values = np.linspace(-2, 2, num=2).reshape(-1, 1)

a2_values = []

for p in p_values:
    a1 = logsiglayer(W1, p, b1)
    a2 = linearlayer(W2, a1, b2)
    a2_values.append(a2)

a2_values = np.array(a2_values).reshape(-1, 1)

plt.plot(p_values, label='p_values', linestyle='--', marker='o', markersize=5)
plt.plot(a2_values, label='a2', linestyle='-', marker='s', markersize=5)
plt.xlabel('p')
plt.ylabel('a2')
plt.title('p vs a2')
plt.legend()
plt.grid(True)
plt.show()


# In[183]:


W1 = np.array([[10, 10]])   # Layer 1

b1 = np.array([[-10],
               [10]])

W2 = np.linspace(-1, 1, num=2).reshape(1, -1)    # Layer 2
b2 = np.array([[0], [0]])

p_values = np.linspace(-2, 2, num=2).reshape(-1, 1)

a2_values = []

for p in p_values:
    a1 = logsiglayer(W1, p_values, b1)
    a2 = linearlayer(W2, a1, b2)
    a2_values.append(a2)

a2_values = np.array(a2_values).reshape(-1, 1)

plt.plot(p_values, label='p_values', linestyle='--', marker='o', markersize=5)
plt.plot(a2_values, label='a2', linestyle='-', marker='s', markersize=5)
plt.xlabel('p')
plt.ylabel('a2')
plt.title('p vs a2')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




