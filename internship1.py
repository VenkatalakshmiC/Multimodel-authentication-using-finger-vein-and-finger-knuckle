# -*- coding: utf-8 -*-
"""
Created on Wed May 24 11:47:12 2023

@author: DELL
"""

# imports
                                    
import pandas as pd
import matplotlib.pyplot as plt


# this allows plots to appear directly in the notebook
%matplotlib inline
# read data into a DataFrame
data = pd.read_csv("advertising.csv")
data.head()
# print the shape of the DataFrame
data.shape
# visualize the relationship between the features and the response using scatterplots
fig, axs = plt.subplots(1,3,sharey=True)
data.plot(kind='scatter',x='TV',y='Sales',ax=axs[0],figsize=(16,8))
data.plot(kind='scatter',x='Radio',y='Sales',ax=axs[1])
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[2])
# create X and y
feature_cols = ['TV']
X = data[feature_cols]
y = data.Sales

# follow the usual sklearn pattern :import ,instantiate,fit
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X,y)

#print intercept and coefficients
print(lm.intercept_)
print(lm.coef_)
# manually calculate the prediction
6.9748214+0.055467*50
# you have to create a DataFrame since the Statsmodels formula interface expects it
X_new =pd.DataFrame({'TV':[50]})
X_new.head()
# use the model to make predictions on a new value
lm.predict(X_new)
# create a DataFrame with the minimum and maximum values of TV
X_new
# make predictions for those x values and store them
preds = lm.predict(X_new)
preds
# first, plot the observed data
data.plot(kind='scatter',x='TV',y='Sales')


# then, plot the least squares line
plt.plot(X_new['TV'], preds, c='red', linewidth=2)