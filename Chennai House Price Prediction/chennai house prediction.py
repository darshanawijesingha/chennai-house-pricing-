#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.linear_model import *
from sklearn.pipeline import *
from sklearn.preprocessing import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import *


# In[4]:


data=pd.read_csv("D:\\data set storing path\\Chennai.csv")
data.describe().T


# In[5]:


data.info()


# In[6]:


label_encoder = LabelEncoder()
data['Location'] = label_encoder.fit_transform(data['Location'])

# Check the DataFrame after conversion
data.head()


# In[7]:


x=data
y=data['Price']


# In[10]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)


# In[11]:


K=np.log(xtrain['Price']/xtrain['Area'])

plt.plot(K,xtrain['Price'],'o')
plt.xlabel('Location decoded value')
plt.ylabel('Price')
plt.title('Bangalore housing price')
plt.show()


# In[12]:


matrix_corr=xtrain.corr()
matrix_corr['Price']


# In[13]:


def na_remove(data):
    data.replace(9,0.5,inplace=True)


# In[14]:


def data_processing(data):
    K=np.log(data['Price']/data['Area'])
    data['Location']=K
    house_feature=data.drop(['Price'],axis=1)
    my_pipeline=Pipeline([('rem',na_remove(house_feature)),
                          ('std',StandardScaler())   
                         ])
    return my_pipeline.fit_transform(house_feature)


# In[16]:


houseprice_train=np.log(xtrain['Price'])
data_train=data_processing(xtrain)


# In[17]:


data_train.shape


# In[18]:


model1=LinearRegression().fit(data_train,houseprice_train)


# In[19]:


model2=Ridge().fit(data_train,houseprice_train)


# In[20]:


model3=DecisionTreeRegressor().fit(data_train,houseprice_train)


# In[21]:


model4=RandomForestRegressor().fit(data_train,houseprice_train)


# In[22]:


model1_pred=model1.predict(data_train)
model2_pred=model2.predict(data_train)
model3_pred=model3.predict(data_train)
model4_pred=model4.predict(data_train)


# In[23]:


# Create a scatter plot of actual vs. predicted values
plt.scatter(houseprice_train, model4_pred, label='Actual vs. Predicted')

# Add a line representing the ideal relationship (y = x)
plt.plot(houseprice_train, houseprice_train, color='red', linestyle='--', label='Ideal')

# Add labels and a legend
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

# Show the plot
plt.show()


# In[24]:


plt.scatter(houseprice_train, model1_pred, label='Actual vs. Predicted')

# Add a line representing the ideal relationship (y = x)
plt.plot(houseprice_train, houseprice_train, color='red', linestyle='--', label='Ideal')

# Add labels and a legend
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

# Show the plot
plt.show()


# In[25]:


# Create a scatter plot of actual vs. predicted values
plt.scatter(houseprice_train, model2_pred, label='Actual vs. Predicted')

# Add a line representing the ideal relationship (y = x)
plt.plot(houseprice_train, houseprice_train, color='red', linestyle='--', label='Ideal')

# Add labels and a legend
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

# Show the plot
plt.show()


# In[26]:


plt.scatter(houseprice_train, model3_pred, label='Actual vs. Predicted')

# Add a line representing the ideal relationship (y = x)
plt.plot(houseprice_train, houseprice_train, color='red', linestyle='--', label='Ideal')

# Add labels and a legend
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

# Show the plot
plt.show()


# In[28]:


houseprice_test=xtest['Price']


# In[29]:


data_test=data_processing(xtest)


# In[30]:


model1_test=np.exp(model1.predict(data_test))
model2_test=np.exp(model2.predict(data_test))
model3_test=np.exp(model3.predict(data_test))
model4_test=np.exp(model4.predict(data_test))


# In[31]:


# Create a scatter plot of actual vs. predicted values
plt.scatter(houseprice_test, model1_test, label='Actual vs. Predicted')

# Add a line representing the ideal relationship (y = x)
plt.plot(houseprice_test, houseprice_test, color='red', linestyle='--', label='Ideal')

# Add labels and a legend
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

# Show the plot
plt.show()


# In[32]:


model1_r2=r2_score(model1_test,houseprice_test)

model2_r2=r2_score(model2_test,houseprice_test)

model3_r2=r2_score(model3_test,houseprice_test)

model4_r2=r2_score(model4_test,houseprice_test)


# In[33]:


print("model1_error:{}\nmodel2_error:{}\nmodel3_error:{}\nmodel4_error:{}".format(model1_r2,model2_r2,model3_r2,model4_r2))


# In[ ]:




