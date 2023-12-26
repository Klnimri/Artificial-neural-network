#!/usr/bin/env python
# coding: utf-8

# In[31]:


import warnings
warnings.filterwarnings("ignore")

#Q1:
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split  

for iter in range(0,5):
    X, y = make_classification(n_samples=100) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y) 
 
    clf = MLPClassifier(max_iter=10).fit(X_train, y_train) 
    clf.predict_proba(X_test[:1]) 
 
    clf.predict(X_test[:5, :]) 
    score = clf.score(X_test, y_test)
    print(f"Iteration {iter + 1}: Score = {score}")


# In[32]:


#Q2
random_state = 42

for iter in range(0, 5):
    X, y = make_classification(n_samples=100, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=random_state)
 
    clf = MLPClassifier(max_iter=10, random_state=random_state).fit(X_train, y_train)
    clf.predict_proba(X_test[:1])
    clf.predict(X_test[:5, :])
 
    score = clf.score(X_test, y_test)
    print(f"Iteration {iter + 1}: Score = {score}")


# In[37]:


#Q3 50:
random_state = 42

for iter in range(0, 5):
    X, y = make_classification(n_samples=100, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=random_state)
 
    clf = MLPClassifier(max_iter=50, random_state=random_state).fit(X_train, y_train)
    clf.predict_proba(X_test[:1])
    clf.predict(X_test[:5, :])
 
    score = clf.score(X_test, y_test)
    print(f"Iteration {iter + 1}: Score = {score}")


# In[38]:


#Q3 100:
random_state = 42

for iter in range(0, 5):
    X, y = make_classification(n_samples=100, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=random_state)
 
    clf = MLPClassifier(max_iter=100, random_state=random_state).fit(X_train, y_train)
    clf.predict_proba(X_test[:1])
    clf.predict(X_test[:5, :])
 
    score = clf.score(X_test, y_test)
    print(f"Iteration {iter + 1}: Score = {score}")


# In[39]:


#Q3 200:
random_state = 42

for iter in range(0, 5):
    X, y = make_classification(n_samples=100, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=random_state)
 
    clf = MLPClassifier(max_iter=200, random_state=random_state).fit(X_train, y_train)
    clf.predict_proba(X_test[:1])
    clf.predict(X_test[:5, :])
 
    score = clf.score(X_test, y_test)
    print(f"Iteration {iter + 1}: Score = {score}")


# In[40]:


#Q3 300:
random_state = 42

for iter in range(0, 5):
    X, y = make_classification(n_samples=100, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=random_state)
 
    clf = MLPClassifier(max_iter=300, random_state=random_state).fit(X_train, y_train)
    clf.predict_proba(X_test[:1])
    clf.predict(X_test[:5, :])
 
    score = clf.score(X_test, y_test)
    print(f"Iteration {iter + 1}: Score = {score}")


# In[ ]:


#Q4 and Q5:
"""
Can not answer these questions because the dataset is not givin and cannot be found
in the internet

"""


# In[ ]:


#Student name: Khalid Nimri
#Student ID: 2140145

