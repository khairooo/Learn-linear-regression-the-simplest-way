# necessary packages 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

# generate random  data-set 
np.random.seed(0)
# generate 100 random numbers with 1d array 
x = np.random.rand(100,1)
y = 2 + 3 * x + np.random.rand(100,1)

# Implemantations with sckit- learn 

# Model initialization 
regression_model = LinearRegression()
# fit the data (train the model)
regression_model.fit(x,y)
# Predict
y_predicted = regression_model.predict(x)
# hint:
'''to visualize the data we can just import the misxtend packages as follow:
>>>from mlxtend.plotting import plot_linear_regression
>>>intercept, slope, corr_coeff = plot_linear_regression(X, y)
>>>plt.show()
'''

# Model evaluation: 
# 1) rmse (root mean squared error)
'''what happen here under the hood is a juste the implimentation of the math equation
>>> np.sqrt(((predictions - targets) ** 2).mean())'''

rmse = mean_squared_error(y,y_predicted)

# 2) R2 ( coefficient of determination) 
'''Explains how much the total variance of the dependent varaible can be reduced 
by  using the least sqaured regression 
mathematically :
R2 = 1 - (SSr/SSt)
where:
# sum of square of residuals
ssr = np.sum((y_pred - y_actual)**2)

#  total sum of squares
sst = np.sum((y_actual - np.mean(y_actual))**2)

# R2 score
r2_score = 1 - (ssr/sst)
'''
r2 = r2_score(y,y_predicted)

# Printing values,slope, intercept, root mean squared error, r2
print('Slope:' ,regression_model.coef_)
print('Intercept:', regression_model.intercept_)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)

# plotting values :
# data points
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()