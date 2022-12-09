#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 12:37:22 2022

@author: judson
"""


#creating classification Tree

#importing packages
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

from dmba import regressionSummary#, exhaustive_search
from dmba import backward_elimination, forward_selection, stepwise_selection
from dmba import AIC_score #, BIC_score, adjusted_r2_score


#Creating a classification tree for Target B 
fundraising_df = pd.read_csv("Fundraising.csv")

#column names
fundraising_df.columns


#setting index
fundraising_df= fundraising_df.set_index('Row Id')

#Exploring data
fundraising_df.head()

#data types
fundraising_df.dtypes


##To find missing values
fundraising_df.isnull().sum()
#no missing values found


#finding correlation
corr = fundraising_df.corr()
corr

corr["TARGET_B"]



#finding best variables
# create a list containing predictors' name
#name predictors
#best variables totalmonths,homeowner dummy, AVGGIFT, INCOME, TIMELAG, None
#predictors = fundraising_df.drop(columns = ["TARGET_D","TARGET_B","Row Id.",])
#predictors = fundraising_df[["totalmonths","homeowner dummy","AVGGIFT","INCOME","TIMELAG"]]

#name predictors
#included all predictors
predictors = fundraising_df.drop(columns = ["TARGET_D","TARGET_B","Row Id.",])
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


#plotting the tree
plotDecisionTree(fullClassTree, feature_names=train_x.columns, rotate = True)



tree = fullClassTree
print('Number of nodes', tree.tree_.node_count)
#Number of nodes 907


# Five-fold cross-validation of the full decision tree classifier
treeClassifier = DecisionTreeClassifier()

scores = cross_val_score(treeClassifier, train_x, train_y, cv=5)
scores


#create a confusion matrix
classificationSummary(train_y,fullClassTree.predict(train_x))
#Confusion Matrix (Accuracy 1.0000)

classificationSummary(valid_y,fullClassTree.predict(valid_x))
#Confusion Matrix (Accuracy 0.5088)

#This shows us that we have omitted variables bias




#creating a new model with best predictors only


#to find best predictors
#using stepwise selection to find best predictors

def train_model(variables):
    if len(variables) == 0: 
        return None
    model = LinearRegression() 
    model.fit(train_x[variables], train_y) 
    return model
    if len(variables) == 0: 
        return None
    model = LinearRegression() 
    model.fit(train_x[variables], train_y) 
    return model

def score_model(model, variables): 
    if len(variables) == 0:
        return AIC_score(train_y, [train_y.mean()] * len(train_y), model, df=1) 
    return AIC_score(train_y, model.predict(train_x[variables]), model)

best_model, best_variables = stepwise_selection(train_x.columns, train_model, score_model, verbose=True)
print(best_variables)

#best variables for the model
#'totalmonths', 'homeowner dummy', 'AVGGIFT', 'INCOME', 'TIMELAG'



#giving only the best predictors as input
predictors = fundraising_df[["totalmonths","homeowner dummy","AVGGIFT","INCOME","TIMELAG"]]


#OUTCOME
outcome = fundraising_df['TARGET_B']



#defining x and y
x = predictors

#changed outcome to string
y = outcome



# partition data; split the data training (60%) vs. validation (40%)
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.4,random_state=1) 
train_x.head()



#create a tree model
fullClassTree = DecisionTreeClassifier()
fullClassTree.fit(train_x, train_y)


#plotting the tree
plotDecisionTree(fullClassTree, feature_names=train_x.columns, rotate = True)


tree = fullClassTree
print('Number of nodes', tree.tree_.node_count)
#Number of nodes 1451



# Five-fold cross-validation of the full decision tree classifier
treeClassifier = DecisionTreeClassifier()

scores = cross_val_score(treeClassifier, train_x, train_y, cv=5)
scores




#Exhaustive grid search to fine tune method parameters
param_grid = {
    'max_depth': [4,5,6,7,8,9], 
    'min_samples_split': [96,97,98,99,100], 
    'min_impurity_decrease': [0, 0.0005, 0.001, 0.005, 0.01], 
}

#gridsearch
gridSearch = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, n_jobs=-1)

#fit the data
gridSearch.fit(train_x, train_y)

#Initial Best score
print('Initial score: ', gridSearch.best_score_)
#Initial score: 0.5502032085561497

#best parameters 
print('Initial parameters: ', gridSearch.best_params_)


# Adapt grid based on result from initial grid search
# run the param_grid block together
param_grid = {
    'max_depth': list(range(1,5)), 
    'min_samples_split': list(range(91,96)), 
    'min_impurity_decrease': [0,0.0009, 0.001, 0.0011], 
}
gridSearch = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, n_jobs=-1)
gridSearch.fit(train_x, train_y)
print('Improved score: ', gridSearch.best_score_)
print('Improved parameters: ', gridSearch.best_params_)


bestClassTree = gridSearch.best_estimator_

plotDecisionTree(bestClassTree, feature_names=train_x.columns)

#model's accuracy
classificationSummary(train_y, bestClassTree.predict(train_x))
##Confusion Matrix (Accuracy 0.5716)

#The model's performance on testing dataset
classificationSummary(valid_y, bestClassTree.predict(valid_x))
#Confusion Matrix (Accuracy 0.5513)





#2)
#creating a neural network to compare the output with classification tree
#already imported the variabless in classification

#imput
input_vars = fundraising_df[["totalmonths","homeowner dummy","AVGGIFT","INCOME","TIMELAG"]]

#output
outcome = fundraising_df['TARGET_B']


# partition data
X = input_vars
Y = outcome
train_X, valid_X, train_Y, valid_Y = train_test_split(X, Y, test_size=0.4, random_state=1)


# train neural network with 2 hidden nodes
network_model = MLPClassifier(hidden_layer_sizes=(6), activation='logistic', solver='adam', random_state=1,max_iter = 5000)
network_model.fit(train_X, train_Y.values)


classes = sorted(Y.unique())
classes


#showing actual y value in Training vs. predicted probabilities
reset_train_Y=train_Y 

reset_train_Y.reset_index(drop=True, inplace=True)
pred=pd.DataFrame(network_model.predict_proba(train_X), columns=classes)


#prediction_X shows the prediction of the neural network model on the training data
prediction_X=pd.concat([reset_train_Y,pred], axis=1)
prediction_X.head()


#showing the weights
for i, (weights, intercepts) in enumerate(zip(network_model.coefs_, network_model.intercepts_)):
    print('Hidden layer' if i == 0 else 'Output layer', '{0[0]} => {0[1]}'.format(weights.shape))
    print(' Intercepts:\n ', intercepts)
    print(' Weights:')
    for weight in weights:
        print(' ', weight)
    print()


# training performance (use idxmax to revert the one-hot-encoding)
classificationSummary(train_Y, network_model.predict(train_X))
#Confusion Matrix (Accuracy 0.5668)

# validation performance
classificationSummary(valid_Y, network_model.predict(valid_X))
#Confusion Matrix (Accuracy 0.5433)





#We can confirm that the classification tree model performs better than
#the neural network model



#Part 2
#Regression Model 
# reduce data frame to the top 2000 rows and select columns for regression analysis
fundraising_regression = fundraising_df.iloc[0:2500]

fundraising_regression.head()
fundraising_regression.shape

#Using Correlation to find the best variables
corr_Target_D = corr["TARGET_D"]
corr_Target_D


#highly correlated predictors
predictors_regression = fundraising_regression.drop(columns=['Row Id.','TARGET_D','TARGET_B'])
print(predictors_regression)


# define outcome/target variable
outcome_regression = fundraising_regression['TARGET_D']
print(outcome_regression)

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





#Regression Model Using Best Predictors

#to find best predictors
# need to run the function as one block
def train_model(variables):
    if len(variables) == 0: 
        return None
    model = LinearRegression() 
    model.fit(train_X[variables], train_Y) 
    return model
    if len(variables) == 0: 
        return None
    model = LinearRegression() 
    model.fit(train_X[variables], train_Y) 
    return model

def score_model(model, variables): 
    if len(variables) == 0:
        return AIC_score(train_Y, [train_Y.mean()] * len(train_Y), model, df=1) 
    return AIC_score(train_Y, model.predict(train_X[variables]), model)

best_model, best_variables = stepwise_selection(train_X.columns, train_model, score_model, verbose=True)
print(best_variables)

#best variables
#['LASTGIFT', 'NUMPROM', 'HV', 'TIMELAG', 'MAXRAMNT', 'RAMNTALL', 'totalmonths', 'zipconvert_4']


#creating a model with only the best variables
predictors_regression = fundraising_regression[['LASTGIFT', 'NUMPROM', 'HV', 'TIMELAG', 'MAXRAMNT', 'RAMNTALL', 'totalmonths', 'zipconvert_4']]
print(predictors_regression)

# define outcome/target variable
outcome_regression = fundraising_regression['TARGET_D']
print(outcome_regression)

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



#printing coefficients
print(pd.DataFrame({'Predictor': X.columns, 'coefficient': fundraising_regression_model.coef_}))




# print performance measures (training data)
regressionSummary(train_Y,fundraising_regression_model.predict(train_X))
#Mean Absolute Error (MAE) : 6.7491

#predictions on a new set
fundraising_pred = fundraising_regression_model.predict(valid_X)
result = pd.DataFrame({'Predicted': fundraising_pred, 'Actual': valid_Y, 'Residual': valid_Y - fundraising_pred})
print(result.head(20))
# print performance measures (validation data)
regressionSummary(valid_Y, fundraising_pred)

#Root Mean Squared Error (RMSE) : 10.6307



