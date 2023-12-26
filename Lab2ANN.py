#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Q1
import math

def single_input_neuron(p, w, b, transfer_function):
    wp = p * w
    n = wp + b
    if transfer_function == 'hardlimit':
        a = 1 if n >= 0 else 0
    elif transfer_function == 'purelin':
        a = n
    elif transfer_function == 'logsig':
        a = 1 / (1 + math.exp(-n))
    elif transfer_function == 'htan':
        a = (math.exp(n) - math.exp(-n)) / (math.exp(n) + math.exp(-n))
    elif transfer_function == 'poslin':
        a = 0 if n < 0 else n
    return a

def choice_menu():
    Choice = int(input("Enter the number of the transfer function: "))
    if  Choice == 1:
        transfer_function = 'hardlimit'
    elif Choice == 2:
        transfer_function = 'purelin'
    elif Choice == 3:
        transfer_function = 'logsig'
    elif Choice == 4:
        transfer_function = 'htan'
    elif Choice == 5:
        transfer_function = 'poslin'
    else:
        print("Not supported transfer function")
        return None
    return transfer_function

p = float(input("Please enter the Input P: "))  
w = float(input("Please enter the Weight W: "))  
b = float(input("Please enter the Bias B: "))

print("Choose the number of the desired transfer function:")
print("1. Hardlimit")
print("2. Purelin")
print("3. Logsig")
print("4. Htan")
print("5. Poslin")

transfer_function = choice_menu() 

if transfer_function is not None:
    output = single_input_neuron(p, w, b, transfer_function)
    print("Output:", output)


# In[8]:


#Q2
import numpy as np

p = np.array([1, 2, 3])
w = np.array([4, 5, 6])
b = -1.5

wp = np.dot(w, p) 
n = wp + b 

def poslin(x):
    return max(0, x)

a = poslin(n)

print("The output of the multiple input neuron using the transfer function \'Poslin\' is: {}".format(a))


# In[ ]:


#Student name: Khalid Nimri
#Student ID: 2140145
#Section TL3

