# yelp-hw
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 18:02:46 2015

@author: liamofarrell
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import statsmodels.formula.api as smf
import seaborn as sns

yelp = pd.read_csv('yelp.csv')

yelp.head()

yelp.groupby('cool').useful.value_counts()
yelp.groupby('cool').funny.value_counts()
yelp.groupby('useful').funny.value_counts()
yelp.groupby('funny').useful.value_counts()

sns.pairplot(yelp, x_vars=['cool','useful','funny'], y_vars='stars', size=6, aspect=0.7, kind='reg')

# The coeficients are how much increasing one of the star, useful or funny rating affects the change in star rating

feature_cols = ['cool','useful', 'funny']
X = yelp[feature_cols]
y = yelp.stars

# instantiate and fit
linreg = LinearRegression()
linreg.fit(X, y)

# print the coefficients
print linreg.intercept_
print linreg.coef_


linreg.predict(50)

def train_test_rmse(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))




feature_cols = ['cool', 'useful', 'funny']
X = yelp[feature_cols]
train_test_rmse(X, y)
