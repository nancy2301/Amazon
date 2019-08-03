# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:35:23 2019

@author: eirik
"""

import os

os.chdir("C:/Users/eirik/OneDrive/Documents/Seattle U classes/Summer 2019/MGMT 5200/Amazon")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing 
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier

master = pd.read_csv("master.csv", sep=',', header=0)

print(master.describe())

cleannum = {"Customer_size": {'Small': 0, 'Mid':1, 'Large':2}}
master.replace(cleannum, inplace = True)

cleannum1 = {"Geo_Code": {'AMER': 6, 'EMEA':5, 'APAC':4, 'JAPN':3, 'CHNA':2, 'GLBL':1, 'GEO-UNCLAIMED':0}}
master.replace(cleannum1, inplace = True)


labels = master["tools_used_not"]
master['tool_used_not[Yes]'] = master['tools_used_not'].astype('category').cat.codes
X = master.drop(['Unnamed: 0', 'Month', 'billing_month', 'Customer_ID', 'tools_used_not', 'Visualize', 'Alert', 'Report', 'Tools', 'tool_used_not[Yes]', ], axis = 1)

clustering_kmeans = KMeans(n_clusters= 6)
master['clusters'] = clustering_kmeans.fit_predict(X)

x = master.drop(['Unnamed: 0', 'billing_month', 'Customer_ID', 'tools_used_not', 'totalBilled', 'Visualize', 'Alert', 'Report', 'Tools', 'tool_used_not[Yes]'], axis = 1)

X_train = x[x['Month'] != 'June-19']
y_train = master[x['Month'] != 'June-19']

X_test = x[x['Month'] == 'June-19']
y_test = master[x['Month'] == 'June-19']

X_train = x.drop(['Month'], axis = 1)

X_test = x.drop(['Month'], axis = 1)

y_train = master['tools_used_not']
y_test = master['tools_used_not']

mlp = MLPClassifier(hidden_layer_sizes=(10), max_iter=1000, activation = 'logistic')  
mlp.fit(X_train, y_train)  

## predict test set 
y_pred = mlp.predict(X_test)  

## confusion matrix
print('')
print('NN Model - one hidden layer - 10 nodes, Logistic')
print(metrics.confusion_matrix(y_test, y_pred))
print("Accuracy", metrics.accuracy_score(y_test, y_pred))

# With activation identity
mlp = MLPClassifier(hidden_layer_sizes=(10), max_iter=1000, activation = 'identity')  
mlp.fit(X_train, y_train)  

## predict test set 
y_pred = mlp.predict(X_test)  

## confusion matrix
print('')
print('NN Model - one hidden layer - 10 nodes, Identity')
print(metrics.confusion_matrix(y_test, y_pred)) 
print("Accuracy", metrics.accuracy_score(y_test, y_pred))


# with activation tanh
mlp = MLPClassifier(hidden_layer_sizes=(10), max_iter=1000, activation = 'tanh')  
mlp.fit(X_train, y_train)  

## predict test set 
y_pred = mlp.predict(X_test)  

## confusion matrix
print('')
print('NN Model - one hidden layer - 10 nodes, tanh')
print(metrics.confusion_matrix(y_test, y_pred)) 
print("Accuracy", metrics.accuracy_score(y_test, y_pred))


# with activation relu
mlp = MLPClassifier(hidden_layer_sizes=(10), max_iter=1000, activation = 'relu')  
mlp.fit(X_train, y_train)  

## predict test set 
y_pred = mlp.predict(X_test)  

## confusion matrix
print('')
print('NN Model - one hidden layer - 10 nodes, relu')
print(metrics.confusion_matrix(y_test, y_pred)) 
print("Accuracy", metrics.accuracy_score(y_test, y_pred))


# with three layers and less nodes
mlp = MLPClassifier(hidden_layer_sizes=(3, 3, 3), max_iter=1000)  
mlp.fit(X_train, y_train)  

## predict test set 
y_pred = mlp.predict(X_test)  

## confusion matrix
print('')
print('NN Model - three hidden layer - 3 nodes')
print(metrics.confusion_matrix(y_test, y_pred)) 
print("Accuracy", metrics.accuracy_score(y_test, y_pred))


from sklearn.naive_bayes import GaussianNB
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print(metrics.confusion_matrix(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))



