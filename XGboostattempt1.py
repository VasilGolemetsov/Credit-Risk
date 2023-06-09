# -*- coding: utf-8 -*-
"""
Created on Sat May 20 15:18:23 2023

@author: mlyil
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb

# import the data
data = pd.read_csv('data2.csv')

#define training and testing sets(should make a function out of this)
train_x,train_y = data.iloc[:700,0:20], data.iloc[:700,20]
test_x, test_y = data.iloc[700:,0:20], data.iloc[700:,20]


#specify the parameters
params_xgb = {
    'objective': 'binary:logistic',  
    'eval_metric': 'logloss',  
    'max_depth': 8,
    'learning_rate': 0.05,
    'gamma': 0,
    'subsample': 1,
    'colsample_bytree': 1,
    'n_jobs': -1
}

# Convert labels to 0 and 1 for the  binary classification
train_y_binary = train_y.replace({1: 0, 2: 1})
test_y_binary = test_y.replace({1: 0, 2: 1})



#fit the model
dtrain = xgb.DMatrix(train_x, label=train_y_binary)
num_round = 1000
bst = xgb.train(params_xgb, dtrain, num_round)
dtest = xgb.DMatrix(test_x, label=test_y_binary)
preds_xgb = bst.predict(dtest)
preds_xgb = pd.DataFrame(preds_xgb, index=test_y.index)

#you can adjust 0.25 as you like
preds_xgb_class = (preds_xgb >= 0.3).astype(int)  

preds_xgb_class = pd.DataFrame({'Actual' : test_y_binary, 'Predicted': preds_xgb_class.squeeze()} ,index=test_y.index)


accuracy = (preds_xgb_class['Actual'] == preds_xgb_class['Predicted']).mean()
print('Accuracy:', accuracy)























