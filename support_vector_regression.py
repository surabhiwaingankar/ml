# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 17:22:26 2023

@author: Lenovo
"""
#SVR

#Data preprocessing

#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values

"""
#splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

"""
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))


#Fitting SVR to the Dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y.ravel())


#Visualizing SVR results
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title("SVM Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

input_value = 6.5
input_array = np.array(input_value).reshape(1, -1)
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(input_array)).reshape(1,-1))