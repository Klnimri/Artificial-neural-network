#!/usr/bin/env python
# coding: utf-8

# In[84]:


import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[59]:


#Q1.
def hardlim(n):
    if n<0:
        return 0
    else:
        return 1
    
def perceptron(W,P,B):
    n = hardlim(np.dot(W,P) + B)
    return n


# In[60]:


#Q2.
#>> a is the predicted output & t is the target output
def PLearningRule(t,wOld,p,bOld,a): 
    e = t - a                       
    wNew = wOld + np.dot(e,p.T)
    bNew = bOld + a
    return wNew , bNew


# In[61]:


#Q3.
perceptron1 = np.array([[-2 , 2]]).T
perceptron2 = np.array([[-2 , 0]]).T
perceptron3 = np.array([[-2 ,-2]]).T
perceptron4 = np.array([[0 , 2]]).T
perceptron5 = np.array([[0 , -2]]).T
perceptron6 = np.array([[2 , 2]]).T
perceptron7 = np.array([[2 , 0]]).T
perceptron8 = np.array([[2 , -2]]).T


# In[62]:


#Q3.
perceptrons = np.array([perceptron1,
                        perceptron2,
                        perceptron3,
                        perceptron4,
                        perceptron5,
                        perceptron6,
                        perceptron7,
                        perceptron8])

T = np.array([1,1,1,1,0,0,0,0])
w = np.round(np.random.randint(0,9,size=(1,2)),3) #1x2
b = np.round(np.random.randint(0,9,size = (1,1)),3) #1x1


# In[68]:


#Q3.
i = 0
count = 0
iter_num = 0

print(f"========{i}th Check========")
print("First W = ", w)
print("First b = ", b)
print(f"===========================")

while count < 8:
    a = perceptron(w, perceptrons[i], b)
    w, b = PLearningRule(T[i], w, perceptrons[i], b, a)
    print(f"{iter_num + 1}th W = ", w)
    print(f"{iter_num + 1}th b = ", b)
    print(f"===========================")

    count = count + 1
    i = i + 1
    iter_num = iter_num + 1


# In[78]:


#Q5.
print("---Checking if the solution for w and b is correct---")
for i in range(n):
    a = perceptron(w, perceptrons[i], b)
    print(f"{i+1}th input = \n{perceptrons[i]} \n Output ={a} \n Target = {T[i]}")
    print("--------------------------------")


# In[83]:


#Q6.
plt.grid() 
marker_size = 80
for i in range(n):
    if T[i] == 1:
        plt.scatter(perceptrons[i][0],
                    perceptrons[i][1],
                    color = "k",
                    s=marker_size)
    else:
        plt.scatter(perceptrons[i][0],
                    perceptrons[i][1],
                    color = "w",
                    edgecolors='black',
                    s=marker_size)


# In[86]:


#Q7.
def decision_boundary(w,b):
    x = np.array([-(b/w[0][0]),0], dtype=float)
    y = np.array([0,-(b/w[0][1])],dtype=float)
    return x , y


# In[111]:


#Q7.
x , y = decision_boundary(w,b)
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.grid()
for i in range(n):
    if T[i] == 1:
        plt.scatter(perceptrons[i][0],
                    perceptrons[i][1],
                    color = "k",
                    s=marker_size);
    else:
        plt.scatter(perceptrons[i][0],
                    perceptrons[i][1],
                    color = "w",
                    edgecolors='black',
                    s=marker_size);
plt.axline(x,y,color = "r");


# In[110]:


#Q8.
fig, ax = plt.subplots()
ax.axline(x,y,color = "r");
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.grid()
#-------------------------------------------------------------------
for i in range(n):
    if T[i] == 1:
        plt.scatter(perceptrons[i][0],
                    perceptrons[i][1],
                    color = "k",
                    s=marker_size);
    else:
        plt.scatter(perceptrons[i][0],
                    perceptrons[i][1],
                    color = "w",
                    edgecolors='black',
                    s=marker_size);
#-------------------------------------------------------------------        
new_point = np.array([-1,2]).T
#-------------------------------------------------------------------
a = perceptron(w, new_point, b)
#-------------------------------------------------------------------
plt.scatter(new_point[0],
            new_point[1],
            color = "r",
            s=marker_size);       
ax.axline(x,y,color = "r");
decision_bound = ax.axline(x,y,color = "r");


# In[ ]:


#Student name: Khalid Nimri
#Student ID: 2140145
#Section: TL3

