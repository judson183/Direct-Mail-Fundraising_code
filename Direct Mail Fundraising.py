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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import dmba
from dmba import plotDecisionTree,classificationSummary
from sklearn import tree

import statsmodels.formula.api as sm	
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
#best variables totalmonths,homeowner dummy, AVGGIFT, INCOME, TIMELAG, None
#predictors = fundraising_df.drop(columns = ["TARGET_D","TARGET_B","Row Id.",])
predictors = fundraising_df[["totalmonths","homeowner dummy","AVGGIFT","INCOME","TIMELAG"]]
predictors.columns
predictors.head()

#OUTCOME
outcome = fundraising_df['TARGET_B']


#defining x and y
x = predictors

#changed outcome to string
y = outcome
#y = outcome.map({1:'Donor',0:'Non_donor'})




# check data type of the predictors
predictors.dtypes 

# partition data; split the data training (60%) vs. validation (40%)
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.4,random_state=1) 
train_x.head()


#create a tree model
fullClassTree = DecisionTreeClassifier()
fullClassTree.fit(train_x, train_y)


plotDecisionTree(fullClassTree, feature_names=train_x.columns, rotate = True)



tree = fullClassTree
print('Number of nodes', tree.tree_.node_count)


#Table 9.6: Exhaustive grid search to fine tune method parameters
# Start with an initial guess for parameters
# run the param_grid block together
param_grid = {
    'max_depth': [5,6,7,8,9,10], 
    'min_samples_split': [96,97,98,99,100], 
    'min_impurity_decrease': [0, 0.0005, 0.001, 0.005, 0.01], 
}

#may take a few seconds to see the results in Console
gridSearch = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, n_jobs=-1)
gridSearch.fit(train_x, train_y)

print('Initial score: ', gridSearch.best_score_)
#Initial score:  0.5486060606060607

print('Initial parameters: ', gridSearch.best_params_)

# Adapt grid based on result from initial grid search
# run the param_grid block together
param_grid = {
    'max_depth': list(range(1,5)), 
    'min_samples_split': list(range(80,90)), 
    'min_impurity_decrease': [0,0.0009, 0.001, 0.0011], 
}
gridSearch = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, n_jobs=-1)
gridSearch.fit(train_x, train_y)
print('Improved score: ', gridSearch.best_score_)
print('Improved parameters: ', gridSearch.best_params_)

bestClassTree = gridSearch.best_estimator_

plotDecisionTree(bestClassTree, feature_names=train_x.columns)

# check model's accuracy
classificationSummary(train_y, bestClassTree.predict(train_x))
##Confusion Matrix (Accuracy 0.5716)
#THe model's peroformance
classificationSummary(valid_y, bestClassTree.predict(valid_x))
#Confusion Matrix (Accuracy 0.5513)




#to find best predictors
# need to run the function as one block
def train_model(variables):
    if len(variables) == 0: 
        return None
    model = LinearRegression() 
    model.fit(train_x[variables], train_y) 
    return model

def score_model(model, variables): 
    if len(variables) == 0:
        return AIC_score(train_y, [train_y.mean()] * len(train_y), model, df=1) 
    return AIC_score(train_y, model.predict(train_x[variables]), model)

best_model, best_variables = forward_selection(train_x.columns, train_model, score_model, verbose=True)
print(best_variables)

#best variables totalmonths,homeowner dummy, AVGGIFT, INCOME, TIMELAG, None






#Part 2
#Regression Model 
# reduce data frame to the top 2000 rows and select columns for regression analysis
fundraising_regression = fundraising_df.iloc[0:2000]

fundraising_regression.head()

#Using Correlation to find the best variables
corr_Target_D = corr["TARGET_D"]
corr_Target_D


#highly correlated predictors
predictors_regression = fundraising_regression.drop(columns=['Row Id.','TARGET_D','TARGET_B'])
print(predictors_regression)


# define outcome/target variable
outcome_regression = fundraising_regression['TARGET_D']
print(outcome)

# check data type of the predictors
predictors_regression.dtypes


#X and Y
X = predictors_regression
Y = outcome_regression

X.shape
Y.shape

# partition data; split the data training (60%) vs. validation (40%)
train_X, valid_X, train_Y, valid_Y = train_test_split(X,Y, test_size=0.4,random_state=1) 
train_X.head()


# check training and validation data sets
data={'Data Set':['train_X', 'valid_X','train_Y','valid_Y'], 'Shape': [train_X.shape, valid_X.shape, train_Y.shape, valid_Y.shape]}
df=pd.DataFrame(data)
print(df)



#build linear regression model using the training data
fundraising_regression_model= LinearRegression()
fundraising_regression_model.fit(train_X, train_Y)


# print coefficients
print(pd.DataFrame({'Predictor': X.columns, 'coefficient': fundraising_regression_model.coef_}))



# print performance measures (training data)
regressionSummary(train_Y,fundraising_regression_model.predict(train_X))


#
# Use predict() to make predictions on a new set
fundraising_pred = fundraising_regression_model.predict(valid_X)
result = pd.DataFrame({'Predicted': fundraising_pred, 'Actual': valid_Y, 'Residual': valid_Y - fundraising_pred})
print(result.head(20))
# print performance measures (validation data)
regressionSummary(valid_Y, fundraising_pred)


