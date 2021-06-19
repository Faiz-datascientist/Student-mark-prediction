#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_csv(r"https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")


# In[4]:


df.head()


# In[6]:


df.tail()


# In[8]:


df.info()


# In[27]:


df.shape


# In[16]:


df.isnull().sum()


# In[10]:


df.describe()


# In[63]:


X=df.drop('Scores',axis=1)
y=df.drop('Hours',axis=1)


# In[64]:


plt.scatter(X,y,color='y')
plt.xlabel("no of hour's")
plt.ylabel("score")
plt.show()


# In[65]:


import sklearn
from sklearn.model_selection import train_test_split


# In[66]:


x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=51)


# In[67]:


print(" shape of x train",x_train.shape)
print(" shape of y train",y_train.shape)
print(" shape of x test",x_test.shape)
print(" shape of y test",y_test.shape)


# In[68]:


from sklearn.linear_model import LinearRegression


# In[69]:


lr=LinearRegression()


# In[70]:


lr.fit(x_train,y_train)


# In[71]:


lr.coef_


# In[72]:


lr.intercept_


# In[74]:


lr.predict([[9.25]])


# In[ ]:




