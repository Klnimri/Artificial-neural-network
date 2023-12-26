#!/usr/bin/env python
# coding: utf-8

# In[105]:


import numpy as np


# In[106]:


#Q1
def purelin(n):
    return n

def FeedForwardLayer(p,w,b):
    n = np.dot(w, p) + b
    a = purelin(n)
    return n


# In[107]:


p = np.array([1, -1, -1]).reshape(-1, 1)
w = np.array([[1, -1, -1], 
              [1, 1, -1]])
b = np.array([[3], 
              [3]])

Result = feedforwardlayer(p, w, b)
print(Result)


# In[108]:


p = np.array([1, 1, -1]).reshape(-1, 1)
w = np.array([[1, -1, -1], 
              [1, 1, -1]])
b = np.array([[3], 
              [3]])

Result = feedforwardlayer(p, w, b)
print(Result)


# In[109]:


p = np.array([-1, -1, -1]).reshape(-1, 1)
w = np.array([[1, -1, -1], 
              [1, 1, -1]])
b = np.array([[3], 
              [3]])

Result = feedforwardlayer(p, w, b)
print(Result)


# In[110]:


p = np.array([-1, 1, -1]).reshape(-1, 1)
w = np.array([[1, -1, -1], 
              [1, 1, -1]])
b = np.array([[3], 
              [3]])

Result = feedforwardlayer(p, w, b)
print(Result)


# In[111]:


#Q3
def recurrent_layer(A1,W,T):
    if (T==0):
        return A1
    else:
        A2=A1
        for i in range(1,T+1):
            n=np.dot(W,A2)
            A=[]
            for item in n:
                a=poslin(item)
                A.append(a)
            A2=A
        return A2
    
def poslin(item):
    if (item<0):
        return 0
    else:
        return item


# In[112]:


w = np.array([[1,-0.5],
              [-0.5,1]])
t = 0
recurrent_layer(Result,w,t)


# In[113]:


w = np.array([[1,-0.5],
              [-0.5,1]])
t = 1
recurrent_layer(Result,w,t)


# In[114]:


w = np.array([[1,-0.5],[-0.5,1]])
t = 2
recurrent_layer(Result,w,t)


# In[115]:


w1 = np.array([[1,-1,-1],
               [1,1,-1]]) #2x3
b1 = np.array([[3],
               [3]]) # 2x1
w2 = np.array([[1,-0.5],
               [-0.5,1]]) # 2x2


# In[116]:


p = np.array([[1,1,1]]).reshape(-1,1) # 1x3
r1 = FeedForwardLayer(p,w1,b1)
print(r1)


# In[117]:


t = 1
recurrent_layer(r1,w2,t)


# In[118]:


t = 2
recurrent_layer(r1,w2,t)


# In[119]:


p = np.array([[1,1,-1]]).reshape(-1,1) # 1x3
r2 = FeedForwardLayer(p,w1,b1)
print(r2)


# In[120]:


t = 1
recurrent_layer(r2,w2,t)


# In[121]:


t = 2
recurrent_layer(r2,w2,t)


# In[122]:


p = np.array([[1,-1,1]]).reshape(-1,1) # 1x3
r2 = FeedForwardLayer(p,w1,b1)
print(r2)


# In[123]:


t = 1
recurrent_layer(r2,w2,t)


# In[124]:


t = 2
recurrent_layer(r2,w2,t)


# In[125]:


p = np.array([[1,-1,-1]]).reshape(-1,1) # 1x3
r2 = FeedForwardLayer(p,w1,b1)
print(r2)


# In[126]:


t = 1
recurrent_layer(r2,w2,t)


# In[127]:


t = 2
recurrent_layer(r2,w2,t)


# In[128]:


p = np.array([[-1,1,1]]).reshape(-1,1) # 1x3
r2 = FeedForwardLayer(p,w1,b1)
print(r2)


# In[129]:


t = 1
recurrent_layer(r2,w2,t)


# In[130]:


t = 2
recurrent_layer(r2,w2,t)


# In[131]:


p = np.array([[-1,1,-1]]).reshape(-1,1) # 1x3
r2 = FeedForwardLayer(p,w1,b1)
print(r2)


# In[132]:


t = 1
recurrent_layer(r2,w2,t)


# In[133]:


t = 2
recurrent_layer(r2,w2,t)


# In[ ]:


#Student name: Khalid Nimri
#Student ID: 2140145
#Section TL3

