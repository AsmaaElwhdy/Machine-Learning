#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries


# In[298]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler ,StandardScaler ,PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error,median_absolute_error


# In[3]:


#reading the data


# In[97]:


path=('D:\machine learning\insurance.csv')
dataset = pd.read_csv(path)


# In[98]:


#analysis the data


# In[99]:


dataset


# In[100]:


dataset.describe()


# In[101]:


dataset.duplicated().sum()


# In[103]:


dataset.drop_duplicates(inplace=True)
dataset


# In[104]:


dataset.duplicated().sum()


# In[105]:


dataset.info()


# In[106]:


dataset.isnull().sum()


# In[107]:


x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


# In[108]:


x


# In[109]:


y


# In[19]:


#preprocessing


# In[112]:


lb=LabelEncoder()
y=lb.fit_transform(y)
x[:,1]=lb.fit_transform(x[:,1])


# In[113]:


lb=LabelEncoder()
y=lb.fit_transform(y)
x[:,4]=lb.fit_transform(x[:,4])


# In[114]:


x


# In[190]:


onehotencoder=ColumnTransformer([('encoder',OneHotEncoder(),[5])],remainder="passthrough")

x=np.array(onehotencoder.fit_transform(x))


# In[191]:


x


# In[201]:


sc_x=MinMaxScaler()
x=sc_x.fit_transform(x)


# In[202]:


x


# In[194]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=42)


# In[195]:


#Linear regression model


# In[310]:


def eval_model(model, x_train, y_train):
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    r2_train = r2_score(y_train, y_pred_train)
    return r2_train


# In[311]:


LinearRegressionModel = LinearRegression()
z=eval_model(LinearRegressionModel, x_train, y_train)
print("linear regression train score is:",LinearRegressionModel.score(x_train,y_train))


# In[313]:


y_pred_test =LinearRegressionModel.predict(x_test)
y_r2=r2_score(y_test, y_pred_test)
print("linear regression predict score is:",y_r2)


# In[314]:


plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred_test, alpha=0.4)
plt.plot([0, 50000], [0, 50000], c='red')
plt.xlabel('y_true')
plt.ylabel('y_pred')
plt.grid(axis='both')
plt.show()


# In[290]:


LinearRegressionModel=LinearRegression()
LinearRegressionModel.fit(x_train,y_train)


# In[291]:


print("linear regression train score is:",LinearRegressionModel.score(x_train,y_train))
print("linear regression test score is:",LinearRegressionModel.score(x_test,y_test))


# In[125]:


y_pred=LinearRegressionModel.predict(x_test)


# In[126]:


print("the tested value for linear regression is",y_test[:10])
print("the predict value for linear regression is",y_pred[:10])


# In[331]:


pf = PolynomialFeatures(degree=3)
X_train_poly = pf.fit_transform(x_train)
X_test_poly = pf.fit_transform(x_test)
nonlinear_reg = LinearRegression()


# In[332]:


R_squ=eval_model(nonlinear_reg, X_train_poly, y_train)
print(R_squ)


# In[333]:


y_pred_test =nonlinear_reg.predict(X_test_poly)
y_r2=r2_score(y_test, y_pred_test)
print(y_r2)


# In[334]:


plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred_test, alpha=0.4)
plt.plot([0, 50000], [0, 50000], c='red')
plt.xlabel('y_true')
plt.ylabel('y_pred')
plt.grid(axis='both')
plt.show()


# In[335]:


#Calculating  Error


# In[336]:


r2Score=r2_score(y_test,y_pred)
print("the r2score is :",r2Score)


# In[131]:


#SVM model


# In[349]:


SVRModel = SVR(C = 0.5 ,epsilon=0.1,kernel = 'linear') 
SVRModel.fit(x_train, y_train)


# In[350]:


print('SVRModel Train Score is : ' , SVRModel.score(x_train, y_train))
print('SVRModel Test Score is : ' , SVRModel.score(x_test, y_test))


# In[351]:


y_pred = SVRModel.predict(x_test)


# In[352]:


print("tested value for SVRModel is:",y_test[:10])
print('Predicted Value for SVRModel is :', y_pred[:10])


# In[253]:


#Calculating  Error


# In[238]:


r2Score=r2_score(y_test,y_pred)
print("the r2score is :",r2Score)


# In[216]:


#Decision Tree Model


# In[354]:


DecisionTreeRegressorModel= DecisionTreeRegressor(max_depth=5,min_samples_split=2,random_state=45)
DecisionTreeRegressorModel.fit(x_train , y_train)


# In[355]:


print('DecisionTreeRegressor Train Score is:',DecisionTreeRegressorModel.score(x_train,y_train))
print('DecisionTreeRegressor test Score is:',DecisionTreeRegressorModel.score(x_test,y_test))


# In[356]:


y_pred=DecisionTreeRegressorModel.predict(x_test)


# In[357]:


print("the tested value for DecisionTreeRegressorModel is",y_test[:10])
print('predicted value for DecisionTreeRegressorModel is: ',y_pred[:10])


# In[358]:


# Calculating  Error


# In[172]:


r2Score=r2_score(y_test,y_pred)
print("the r2score is :",r2Score)


# In[ ]:




