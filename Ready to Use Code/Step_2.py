#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 18:49:45 2019

@author: Eirik Fosnaes​, Jillian Pflugrath​, Nancy Jain​, Yifan Xiang​, Ziyu Jin
"""

#Set your working directory here
#import os
#os.chdir("C:/Users/eirik/OneDrive/Documents/Seattle U classes/Summer 2019/MGMT 5200/Amazon")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import patsy
import statsmodels.api as sm


def getFile(name):
    name = name + ".csv"
    master = pd.read_csv(name, sep=',', header=0)
    
    return (master)


def cleanData(master):
    # Set cutomer_size to codes
    cleannum = {"Customer_size": {'Small': 0, 'Mid':1, 'Large':2}}
    master.replace(cleannum, inplace = True)
    
    # Set geo_code ro codes
    cleannum1 = {"Geo_Code": {'AMER': 5, 'EMEA':4, 'APAC':3, 'JAPN':2, 'CHNA':1, 'GLBL':0}}
    master.replace(cleannum1, inplace = True)

    return(master)

    
def prepare(master):
    master['tool_used_not[Yes]'] = master['tools_used_not'].astype('category').cat.codes
    X = master.drop(['Unnamed: 0', 'Month', 'Customer_ID', 'Customer_size', 'tools_used_not', 'Visualize', 'Alert', 'Report', 'Tools', 'tool_used_not[Yes]'], axis = 1)
    scaler = StandardScaler()  
    scaler.fit(X)
    X = scaler.transform(X)
    
    return(X)
    
def model_Cluster(X, X_new): 
    clustering_kmeans = KMeans(n_clusters= 5, random_state = 74)
    clustering_kmeans.fit(X)

    cluster_pred = clustering_kmeans.predict(X_new)
    
    return(cluster_pred)
    
def model_yes_no(master, new):   
    # Prepare data
    x = master.drop(['Unnamed: 0', 'Customer_ID', 'tools_used_not', 'Visualize', 'Alert', 'Report', 'Tools', 'tool_used_not[Yes]'], axis = 1)
    x_new = new.drop(['Unnamed: 0', 'Customer_ID', 'tools_used_not', 'Visualize', 'Alert', 'Report', 'Tools', 'tool_used_not[Yes]'], axis = 1)
    
    # Get training dataset
    X_train = x[x['Month'] != 'June-19']
    y_train = master[master['Month'] != 'June-19']
    
    X_train = X_train.drop(['Month'], axis = 1)
    
    y_train = y_train['tools_used_not']
    
    x_new = x_new.drop(['Month'], axis = 1)

    # Train the model 
    mlp = MLPClassifier(hidden_layer_sizes=(8), max_iter=1000, activation = 'identity', random_state = 9)  
    mlp.fit(X_train, y_train)  

    # Predict 
    new_pred = mlp.predict(x_new)  
    
    return(new_pred)

def remove_no_master(All):

    # Filter out the cutomers not using any tools
    All = All[All['tools_used_not'] == 'Yes']
    
    return(All)
    
def remove_no_new(All):

    # Filter out the cutomers not using any tools
    All = All[All['Yes_No_Prediction'] == 'Yes']
    
    return(All)


def model_Visualize(All, new):
    
    y_vis = All['Visualize']

    # Logit Model
    interaction = "Visualize ~   customer_age + totalBilled+clusters"
    y,XX = patsy.dmatrices(interaction, All, return_type = "dataframe")
    X_new = new[['customer_age','totalBilled','clusters']]  
    
    # Seperate training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(XX, y_vis, test_size=0.30,random_state=9)
    num_col_names = ['totalBilled','customer_age']   ## scale only numeric variable
    scaler = StandardScaler().fit(X_train[num_col_names].values)
    X_train[num_col_names] = scaler.transform(X_train[num_col_names].values)
    X_new[num_col_names] = scaler.transform(X_new[num_col_names].values)
   

    # Run the model
    Logit = sm.MNLogit(y_train, X_train).fit_regularized()
    X_new["Intercept"] = 1
    
    print(X_new.info())
    print(X_train.info())
    y_pred_prob = Logit.predict(X_new)
    y_pred = y_vis.astype('category').cat.categories[y_pred_prob.idxmax(axis=1)]
    
    
    return(y_pred)
    
def model_Alert(All, new):
    
    # Prepare data   
    y_vis = All['Alert']
    X = All[['Geo_Code','ProductsUsed','customer_age','totalBilled','clusters','Customer_size']]    
    X_new = new[['Geo_Code','ProductsUsed','customer_age','totalBilled','clusters','Customer_size']]  
    
    # Get training data   
    X = pd.get_dummies(X,drop_first=True,prefix_sep='_')
    X_train, X_test, y_train, y_test = train_test_split(X, y_vis, test_size=0.30,random_state=9)
    
    # Train the model
    clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10, max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=100, splitter='best')
    clf = clf.fit(X_train, y_train)

    # Run the model
    y_pred = clf.predict(X_new)
    
    
    return(y_pred)

def model_Report(All, new):
    # Prepare data   
    y_vis = All['Report']
    X = All[['Geo_Code','ProductsUsed','customer_age','totalBilled','clusters','Customer_size']]    
    X_new = new[['Geo_Code','ProductsUsed','customer_age','totalBilled','clusters','Customer_size']]  
    

    # Get training data             
    X = pd.get_dummies(X,drop_first=True,prefix_sep='_')
    X_train, X_test, y_train, y_test = train_test_split(X, y_vis, test_size=0.30,random_state=9)
    
    # Train the model
    clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10, max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=100, splitter='best')
    clf = clf.fit(X_train, y_train)
    
    # Run the model
    y_pred = clf.predict(X_new)     
    
    
    return(y_pred)



def Main():
    # Get the training data and the new data
    master = getFile("master")
    new = getFile("new")   
    
    # Clean datasets
    master = cleanData(master)
    new = cleanData(new)
       
    # Add clusters to datasets
    master["clusters"] = model_Cluster(prepare(master),prepare(master))
    new["clusters"] = model_Cluster(prepare(new),prepare(new))

    # Predict whether customers will use any of the products
    new["Yes_No_Prediction"] = model_yes_no(master, new)
    
    # Slect customers that are predicted to buy 
    master_yes = remove_no_master(master)
    new_yes = remove_no_new(new)
    
    # If yes, predict whether customers will use:
    # Visualize
    new_yes["Visulaize_Prediction"] = model_Visualize(master_yes, new_yes)
    # Alert
    new_yes["Alert_Prediction"] = model_Alert(master_yes, new_yes)
    # Report
    new_yes["Report_Prediction"] = model_Report(master_yes, new_yes)
    
    new.to_csv( "new_yes_no_result.csv", index=False, encoding='utf-8-sig')
    new_yes.to_csv( "new_product_result.csv", index=False, encoding='utf-8-sig')
    
    


if __name__ == "__main__":
    Main()