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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import statsmodels.api as sm
import graphviz #optional –needed to render a tree model into graph
from sklearn.preprocessing import StandardScaler

master = pd.read_csv("master.csv", sep=',', header=0)

print(master.describe())

cleannum = {"Customer_size": {'Small': 0, 'Mid':1, 'Large':2}}
master.replace(cleannum, inplace = True)

cleannum1 = {"Geo_Code": {'AMER': 5, 'EMEA':4, 'APAC':3, 'JAPN':2, 'CHNA':1, 'GLBL':0}}
master.replace(cleannum1, inplace = True)


labels = master["tools_used_not"]
master['tool_used_not[Yes]'] = master['tools_used_not'].astype('category').cat.codes
X = master.drop(['Unnamed: 0', 'Month', 'billing_month', 'Customer_ID', 'Customer_size', 'tools_used_not', 'Visualize', 'Alert', 'Report', 'Tools', 'tool_used_not[Yes]'], axis = 1)

scaler = StandardScaler()  
scaler.fit(X)
X = scaler.transform(X) 

clustering_kmeans = KMeans(n_clusters= 5, random_state = 74)
master['clusters'] = clustering_kmeans.fit_predict(X)

#master.to_csv('master_clust5.csv')

grouped_data = master.groupby(['clusters'])

grouped_data['Customer_size'].aggregate(np.mean).reset_index()
grouped_data['ProductsUsed'].aggregate(np.mean).reset_index()
grouped_data['totalBilled'].aggregate(np.mean).reset_index()
grouped_data['customer_age'].aggregate(np.mean).reset_index()

x = master.drop(['Unnamed: 0', 'billing_month', 'Customer_ID', 'tools_used_not', 'Visualize', 'Alert', 'Report', 'Tools', 'tool_used_not[Yes]'], axis = 1)

X_train = x[x['Month'] != 'June-19']
y_train = master[master['Month'] != 'June-19']

X_test = x[x['Month'] == 'June-19']
y_test = master[master['Month'] == 'June-19']

X_train = X_train.drop(['Month'], axis = 1)

X_test = X_test.drop(['Month'], axis = 1)

y_train = y_train['tools_used_not']
y_test = y_test['tools_used_not']


#mlp = MLPClassifier(hidden_layer_sizes=(10), max_iter=1000, activation = 'logistic', random_state = 109)  
#mlp.fit(X_train, y_train)  

## predict test set 
#y_pred = mlp.predict(X_test)  

## confusion matrix
#print('')
#print('NN Model - one hidden layer - 10 nodes, Logistic')
#print(metrics.confusion_matrix(y_test, y_pred))

#print("Neural Network Test Accuracy:",metrics.accuracy_score(y_test, y_pred))



# With activation identity
mlp = MLPClassifier(hidden_layer_sizes=(8), max_iter=1000, activation = 'identity', random_state = 9)  
mlp.fit(X_train, y_train)  

## predict test set 
y_pred = mlp.predict(X_test)  

## confusion matrix
print('')
print('NN Model - one hidden layer - 8 nodes, Identity')
print(metrics.confusion_matrix(y_test, y_pred)) 
print("Neural Network Test Accuracy:",metrics.accuracy_score(y_test, y_pred))
#print("Recall:", metrics.recall_score(y_test,y_pred,average = 'micro'))


# with activation tanh
mlp = MLPClassifier(hidden_layer_sizes=(10), max_iter=1000, activation = 'tanh', random_state = 9)  
mlp.fit(X_train, y_train)  

## predict test set 
y_pred = mlp.predict(X_test)  

X_test.to_csv('X_test.csv')

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



# with three layers and less nodes
mlp = MLPClassifier(hidden_layer_sizes=(3, 6), max_iter=1000, activation = 'logistic', seed = 10)  
mlp.fit(X_train, y_train)  

## predict test set 
y_pred = mlp.predict(X_test)  

## confusion matrix
print('')
print('NN Model - three hidden layer - 3 nodes')
print(metrics.confusion_matrix(y_test, y_pred)) 
print("Accuracy", metrics.accuracy_score(y_test, y_pred))


# with three layers and less nodes
mlp = MLPClassifier(hidden_layer_sizes=(3, 3), max_iter=1000)  
mlp.fit(X_train, y_train)  

## predict test set 
y_pred = mlp.predict(X_test)  

## confusion matrix
print('')
print('NN Model - three hidden layer - 3 nodes')
print(metrics.confusion_matrix(y_test, y_pred)) 
print("Accuracy", metrics.accuracy_score(y_test, y_pred))



# With activation identity
mlp = MLPClassifier(hidden_layer_sizes=(8), max_iter=1000, activation = 'identity')  
mlp.fit(X_train, y_train)  

## predict test set 
y_pred = mlp.predict(X_test)  

## confusion matrix
print('')
print('NN Model - one hidden layer - 8 nodes, Identity')
print(metrics.confusion_matrix(y_test, y_pred)) 



# With activation identity
mlp = MLPClassifier(hidden_layer_sizes=(10), max_iter=1000, activation = 'identity')  
mlp.fit(X_train, y_train)  

## predict test set 
y_pred = mlp.predict(X_test)  

## confusion matrix
print('')
print('NN Model - one hidden layer - 7 nodes, Identity')
print(metrics.confusion_matrix(y_test, y_pred)) 

clf = tree.DecisionTreeClassifier(class_weight = None, criterion = 'gini', max_depth = 10, max_features = None, \
max_leaf_nodes = None, min_samples_leaf = 10, min_samples_split = 2, min_weight_fraction_leaf = 0.0, presort = False, \
random_state = 109, splitter = 'best')
clf = clf.fit(X_train, y_train)

# export estimated tree into dot graphic file
dot_data = tree.export_graphviz(clf, out_file='Dtree.dot', feature_names=X.columns)

y_pred = clf.predict(X_test)  
print(metrics.confusion_matrix(y_test, y_pred))
print("Accuracy", metrics.accuracy_score(y_test, y_pred))













