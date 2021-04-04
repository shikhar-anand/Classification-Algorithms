#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install openml


# In[2]:


import pandas as pd
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support


# In[ ]:





# In[3]:


iris = fetch_openml(name='iris')


# In[4]:


iris.data.shape


# In[5]:


iris.target.shape


# In[6]:


np.unique(iris.target)


# In[7]:


iris.DESCR


# In[8]:


iris.details


# In[9]:


iris


# ### Converting numpy array to pandas dataframe

# In[10]:


iris.data.shape


# In[11]:


iris.target.shape


# In[12]:


X_df = pd.DataFrame(data=iris.data,    # values
    index=np.array(range(1,151)),    # 1st column as index
    columns=np.array(range(1,5)))  # 1st row as the column names


# In[13]:


y_df = pd.DataFrame(data=iris.target,    # values
    index=np.array(range(1,151)),    # 1st column as index
    columns=np.array(range(1,2)))  # 1st row as the column names


# In[14]:


X_df


# In[15]:


y_df


# In[ ]:





# ### Training the model:
# 

# In[16]:


neigh = KNeighborsClassifier(n_neighbors = 4)
neigh.fit(X_df,y_df)


# ### Performing some test operation the model:

# In[17]:


testSet = [[1.4, 3.6, 3.4, 1.2]]
test = pd.DataFrame(testSet)
print(test)
print("predicted:",neigh.predict(test))
print("neighbors",neigh.kneighbors(test))


# ### KNN using train and test sets ,and finding different parameters and analysis of output.

# In[ ]:





# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2)
# test_size determines the percentage of test data you want here
# train=80% and test=20% data is randomly split


# In[19]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# ### Changing categorical iris target data to numeric

# In[20]:


y_df[1].unique()


# In[21]:


y_df[1] = y_df[1].replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [1, 2, 3])


# In[22]:


type(y_df)


# In[23]:


rskf = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 10, random_state = 2)


# In[24]:


i = 0;
accuracy_lst = []
precision_lst = []
recall_lst = []
f1_lst = []
precision_recall_fscore_lst = []

for train_index, test_index in rskf.split(X_df, y_df):
    i = i+1;
    X_train, X_test = X_df.iloc[train_index], X_df.iloc[test_index]
    y_train, y_test = y_df.iloc[train_index], y_df.iloc[test_index]
    neigh.fit(X_df.iloc[train_index], y_df.iloc[train_index])
    y_pred = neigh.predict(X_df.iloc[test_index])
    
    accuracy_value = accuracy_score(y_df.iloc[test_index], y_pred)
    accuracy_lst.append(accuracy_value)
    
    precision_value = precision_score(y_df.iloc[test_index], y_pred, average = 'macro')
    precision_lst.append(precision_value)
    
    recall_value = recall_score(y_df.iloc[test_index], y_pred, average = 'macro')
    recall_lst.append(recall_value)
    
    f1_value = f1_score(y_df.iloc[test_index], y_pred, average = 'macro')
    f1_lst.append(f1_value)
    
    precision_recall_fscore_value = precision_recall_fscore_support(y_df.iloc[test_index], y_pred, average = 'macro')
    precision_recall_fscore_lst.append(precision_recall_fscore_value)


# In[25]:


print('Metrics using Average = macro\nAccuracy: %f, Precision: %f, Recall: %f, f1_score: %f' 
      %(np.mean(accuracy_lst) * 100, 
        np.mean(precision_lst) * 100, 
        np.mean(recall_lst) * 100, 
        np.mean(f1_lst) * 100))

