import os
import numpy as np
import pandas as pd
import statsmodels.api as st
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import patsy
import graphviz

os.chdir("/Users/xiangyifan/Desktop/SU/MSBA/19RQ/MGMT5200/Project/AMAZOn")
All = pd.read_csv("All.csv", sep = ',', header = 0)


# divide df into testing df and training df

All_train = All[All.Month != 'June-19']
All_test = All[All.Month == 'June-19']

y_train = All_train["Tools"]
y_test = All_test['Tools']


## Decision Tree

col_names = ['Customer_size', 'Geo_Code', 'ProductsUsed', 'totalBilled', 'customer_age']     ## variable selected
tree_train = All_train[col_names]
tree_test = All_test[col_names]

X_train = pd.get_dummies(tree_train,drop_first=True,prefix_sep='_')
X_test = pd.get_dummies(tree_test,drop_first=True,prefix_sep='_')

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=109)


clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10, max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=100, splitter='best')


 
clf = clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)   ## predict train set
y_pred_test = clf.predict(X_test)     ## predict test set
print(metrics.confusion_matrix(y_test, y_pred_test))
print("Decision Tree Train Accuracy:",metrics.accuracy_score(y_train, y_pred_train))
print("Decision Tree Test Accuracy:",metrics.accuracy_score(y_test, y_pred_test))
print("F score:", metrics.f1_score(y_test,y_pred_test,average = 'weighted'))
print("Precision: ", metrics.precision_score(y_test,y_pred_test,average = 'weighted'))
print("Recall:", metrics.recall_score(y_test,y_pred_test,average = 'weighted'))

# export estimated tree into dot graphic file
dot_data = tree.export_graphviz(clf, out_file='Dtree.dot',filled=True,rounded = True, feature_names=X_train.columns)


# MNLogit Model
interaction2 = "Tools ~ Customer_size + Geo_Code + ProductsUsed + totalBilled + customer_age "

y_train,XX2train = patsy.dmatrices(interaction2, All_train, return_type = "dataframe")

y_test,XX2test = patsy.dmatrices(interaction2, All_test, return_type = "dataframe")

#X_train, X_test, y_train, y_test = train_test_split(XX2, y, test_size=0.30,random_state=109)
num_col_names = ['ProductsUsed','totalBilled','customer_age']   ## scale only numeric variable
scaler = StandardScaler().fit(XX2train[num_col_names].values)
X_train[num_col_names] = scaler.transform(XX2train[num_col_names].values)
X_test[num_col_names] = scaler.transform(XX2test[num_col_names].values)
Logit = st.MNLogit(y_train, X_train).fit_regularized()

y = All['Tools']
print(Logit.summary())
y_prob_train = Logit.predict(X_train)
y_pred_train = y.astype('category').cat.categories[y_prob_train.idxmax(axis=1)]
y_prob_test = Logit.predict(X_test)
y_pred_test = y.astype('category').cat.categories[y_prob_test.idxmax(axis=1)]

print(metrics.confusion_matrix(y_test, y_pred_test))
print("MNLogit Train Accuracy:",metrics.accuracy_score(y_train, y_pred_train))
print("MNLogit Test Accuracy:",metrics.accuracy_score(y_test, y_pred_test))
print("F score:", metrics.f1_score(y_test,y_pred_test,average = 'weighted'))
print("Precision: ", metrics.precision_score(y_test,y_pred_test,average = 'weighted'))
print("Recall:", metrics.recall_score(y_test,y_pred_test,average = 'weighted'))
