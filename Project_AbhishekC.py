# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:55:55 2023

@author: abhis
"""

import pandas as pd
import numpy as np
import math 
#from pandas.plotting import scatter_matrix, parallel_coordinates
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import dmba
from dmba import plotDecisionTree,regressionSummary
from sklearn import tree
#import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from dmba import plotDecisionTree, classificationSummary, regressionSummary


#from sklearn import metrics
#pd.set_option('display.float_format', lambda x: '%.5f' % x)

#import warnings
#warnings.filterwarnings('ignore') 

c_df = pd.read_csv(r'C:\Users\abhis\catelog.csv')


## !. Cleaning the data and checking the quality for model creation and learning.


## List the variable types of each column.
Vtype = dict(c_df.dtypes) 
Vtype
c_df.columns

##  List the dimensions of the data frame.
c_df.shape
row,columns =c_df.shape
print("No of rows,",row)
print("No of columns,",columns)

## Checking for null values

c_df.isna().any()
c_df.isnull().sum()


## Filling missing values with median.

m_last_update_days_ago = c_df['last_update_days_ago'].mean()
c_df.last_update_days_ago = c_df.last_update_days_ago.fillna(value=m_last_update_days_ago)

c_df['1st_update_days_ago'].fillna(c_df['1st_update_days_ago'].mean(), inplace=True)
#c_df['1st_update_days_ago'].fillna(c_df['1st_update_days_ago'].median(), inplace=True)

## For web orders we cannotsimply replace it with any value, so we can either 
## drop rows or crate a missing value and create dummies.
#c_df["Web order"] = c_df["Web order"].fillna("Missing")
c_df=c_df.dropna(subset=["Web order"])
c_df.isnull().sum()



print(c_df['Purchase'].value_counts())
c_df.hist(bins=10, figsize=(12,12))
sns.catplot(data=c_df, kind="bar", x="Purchase", y="Freq", hue="Gender=male")


## simple heatmap of correlations (without values)

corr = c_df.corr()
print(corr)



#import seaborn as sns
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
# Change the colormap to a divergent scale and fix the range of the colormap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, vmin=-1, vmax=1, cmap="RdBu")# vmin & vmax set the limit of the colormap
# Include information about values (example demonstrate how to control the size of the plot
# run the block together
fig, ax = plt.subplots()
fig.set_size_inches(11, 7)




## Getting dummies if necessary, here we have all the columns in numerical format so we do
##  not need dummies, hence creating target and predictor variables.

X = c_df.drop(columns=['Purchase'],axis=1)
#X = pd.get_dummies(c_df, drop_first=True) 
X.info()

c_df['Purchase'].value_counts()
y = c_df['Purchase']


#c_df = c_df.drop(['sequence_number', 'Purchase'], axis = 1)
#c_df.head(10)

from sklearn.model_selection import train_test_split
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.3, random_state=1)
data={'Data Set':['train_X', 'valid_X','train_y','valid_y'], 'Shape': [train_X.shape, valid_X.shape, train_y.shape, valid_y.shape]}
df=pd.DataFrame(data)
df

print(train_X.isnull().any()) 
print(train_y.isnull().any())


###
#create a tree model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
fullClassTree = DecisionTreeClassifier()
fullClassTree.fit(train_X, train_y)



plotDecisionTree(fullClassTree, feature_names=train_X.columns)
text_representation = tree.export_text(fullClassTree, feature_names=list (train_X.columns))
print(text_representation)


tree = fullClassTree
print('Number of nodes', tree.tree_.node_count)

#Table 9.3 confusion matrix to measure a model's accuracy
# accuracy on the training data
classificationSummary(train_y, fullClassTree.predict(train_X))

#accuracy on the validation data
classificationSummary(valid_y, fullClassTree.predict(valid_X))


# Five-fold cross-validation of the full decision tree classifier
treeClassifier = DecisionTreeClassifier()
from sklearn.model_selection import cross_val_score

scores = cross_val_score(treeClassifier, train_X, train_y, cv=5)
scores

#f'{acc:.3f}' -- f-string provide a simple way to include the vale of python expression inside string
# value=80
#f'The value is {value}.'
#The value is 80.
#pring 3 decimal places {:.3f}
print('Accuracy scores of each fold: ', [f'{acc:.3f}' for acc in scores])
print('Accuracy scores of each fold: ', scores)
# range of two times of standard deviation
print(f'Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})')
# range of standard deviation
print(f'Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})')

from sklearn import tree

#figure 9.12
smallClassTree = DecisionTreeClassifier(max_depth=30, min_samples_split=20, min_impurity_decrease=0.01)
smallClassTree.fit(train_X, train_y)
plotDecisionTree(smallClassTree, feature_names=train_X.columns)
text_representation = tree.export_text(smallClassTree, feature_names=train_X.columns)
print(text_representation)

#Table 9.5 check accuracy on the training and validation data
classificationSummary(train_y, smallClassTree.predict(train_X))
classificationSummary(valid_y, smallClassTree.predict(valid_X))

#Table 9.6: Exhaustive grid search to fine tune method parameters
# Start with an initial guess for parameters
# run the param_grid block together
param_grid = {
    'max_depth': [10, 20, 30, 40], 
    'min_samples_split': [20, 40, 60, 80, 100], 
    'min_impurity_decrease': [0, 0.0005, 0.001, 0.005, 0.01], 
}

#may take a few seconds to see the results in Console
gridSearch = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, n_jobs=-1)
gridSearch.fit(train_X, train_y)

print('Initial score: ', gridSearch.best_score_)
print('Initial parameters: ', gridSearch.best_params_)

# Adapt grid based on result from initial grid search
# run the param_grid block together
param_grid = {
    'max_depth': [15, 20, 25, 30, 35] , 
    'min_samples_split': [15, 18, 20, 25, 30], 
    'min_impurity_decrease': [0.0009, 0.001, 0.0011], 
}
gridSearch = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, n_jobs=-1)
gridSearch.fit(train_X, train_y)
print('Improved score: ', gridSearch.best_score_)
print('Improved parameters: ', gridSearch.best_params_)

bestClassTree = gridSearch.best_estimator_

plotDecisionTree(bestClassTree, feature_names=train_X.columns)
text_representation = tree.export_text(bestClassTree, feature_names=list(train_X.columns))
print(text_representation)

# check model's accuracy
classificationSummary(train_y, bestClassTree.predict(train_X))
classificationSummary(valid_y, bestClassTree.predict(valid_X))



## Neural Ntwork
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

clf = MLPClassifier(hidden_layer_sizes=(4), activation='logistic', solver='lbfgs', random_state=1, max_iter=500)
clf.fit(train_X, train_y)

#show the weights, run the block together
for i, (weights, intercepts) in enumerate(zip(clf.coefs_, clf.intercepts_)):
    print('Hidden layer' if i == 0 else 'Output layer', '{0[0]} => {0[1]}'.format(weights.shape))
    print(' Intercepts:\n ', intercepts)
    print(' Weights:')
    for weight in weights:
        print(' ', weight)
    print()

# Gridsearch
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

# train neural network with 4 hidden nodes
clf = MLPClassifier(hidden_layer_sizes=(4), activation='logistic', solver='lbfgs', random_state=1, max_iter=500)
clf.fit(train_X, train_y.values)
#clf.fit(train_X, train_y)

classificationSummary(train_y, clf.predict(train_X))

# validation performance
classificationSummary(valid_y, clf.predict(valid_X)).__bool__()

# define the numbers of hidden layer sizes you want to try
param_grid = {
    'hidden_layer_sizes': [(1), (2), (3), (4), (5)], 
}

#run the two lines together as one block
gridSearch = GridSearchCV(MLPClassifier(activation='logistic', solver='lbfgs', random_state=1, max_iter=1000), 
                          param_grid, cv=5, n_jobs=-1, return_train_score=True)
gridSearch.fit(train_X, train_y)
print('Initial score: ', gridSearch.best_score_)
print('Best parameters: ', gridSearch.best_params_)

param_grid2 = {'hidden_layer_sizes': [(3,), (4, ), (5, )]
               }
gridSearch1 = GridSearchCV(MLPClassifier(activation='logistic', solver='lbfgs', random_state=1, max_iter=1000), 
                          param_grid2, cv=5, n_jobs=-1, return_train_score=True)
gridSearch1.fit(train_X, train_y)
print('Best score: ', gridSearch1.best_score_)
print('Best parameters: ', gridSearch1.best_params_)

display=['param_hidden_layer_sizes', 'mean_test_score', 'std_test_score']
print(pd.DataFrame(gridSearch.cv_results_)[display])
#gridSearch.cv_results_

pd.DataFrame(gridSearch.cv_results_)[display].plot(x='param_hidden_layer_sizes', y='mean_test_score', yerr='std_test_score', ylim=(0.8, 0.9))
plt.show()

## Evaluating model performance using training performance and validation performance.
# training performance (use idxmax to revert the one-hot-encoding)
classificationSummary(train_y, gridSearch1.predict(train_X))

# validation performance
classificationSummary(valid_y, gridSearch1.predict(valid_X)).__bool__()

