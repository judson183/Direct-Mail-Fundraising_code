#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 12:37:22 2022

@author: judson
"""


#creating classification Tree

#importing packages
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import dmba
from dmba import plotDecisionTree,regressionSummary
from sklearn import tree

import dmba
from dmba import classificationSummary
from dmba import regressionSummary#, exhaustive_search
from dmba import backward_elimination, forward_selection, stepwise_selection
from dmba import AIC_score #, BIC_score, adjusted_r2_score


#Target B 
fundraising_df = pd.read_csv("Fundraising.csv")

fundraising_df.columns


#setting index
fundraising_df= fundraising_df.set_index('Row Id')

#Exploring data
fundraising_df.head()

#data types
fundraising_df.dtypes


##To find missing values
fundraising_df.isnull().sum()

#to create decision tree model 


#finding correlation
corr = fundraising_df.corr()
corr

corr["TARGET_B"]



#finding best variables
# create a list containing predictors' name
#name predictors
predictors = fundraising_df.drop(columns = ["TARGET_D","TARGET_B","Row Id."])
predictors.columns
predictors.head()

#OUTCOME
outcome = fundraising_df['TARGET_B']


#defining x and y
x = predictors
y = outcome


# check data type of the predictors
predictors.dtypes 

# partition data; split the data training (60%) vs. validation (40%)
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.4,random_state=1) 
train_x.head()


# user grid search to find optimized tree
# run param_grid block together
param_grid = {
    'max_depth': [5, 10, 15, 20, 25], 
    'min_impurity_decrease': [0, 0.001, 0.005, 0.01], 
    'min_samples_split': [10, 20, 30, 40, 50], 
}
gridSearch = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5, n_jobs=-1)
gridSearch.fit(train_x, train_y)
print('Initial parameters: ', gridSearch.best_params_)


# run param_grid block together
param_grid = {
    'max_depth': [1,2,3,4,5], 
    'min_impurity_decrease': [0, 0.001, 0.002, 0.003, 0.004, 0.005], 
    'min_samples_split': [10,11,12,13,14,], 
}
gridSearch = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5, n_jobs=-1)
gridSearch.fit(train_x, train_y)
print('Improved parameters: ', gridSearch.best_params_)

regTree = gridSearch.best_estimator_

#regression tree performance
regressionSummary(train_y, regTree.predict(train_x))
regressionSummary(valid_y, regTree.predict(valid_x))

plotDecisionTree(regTree, feature_names=train_x.columns)
plotDecisionTree(regTree, feature_names=train_x.columns, rotate=True)


