import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier 
from sklearn import metrics
from sklearn import tree
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


data = pd.read_csv("master.csv", sep=',', header = 0)

data['tools_used_not']=data['tools_used_not'].astype('category').cat.codes

data['Customer_size']=data['Customer_size'].astype('category').cat.codes

data['Geo_Code']=data['Geo_Code'].astype('category').cat.codes





###### Decision Tree 1 #######
print("###Tree 1###")
y = data['tools_used_not']
X = data[['Customer_size', 'Geo_Code', 'ProductsUsed', 'totalBilled', 'customer_age']] 
  
clf = tree.DecisionTreeClassifier(class_weight = None, criterion = 'gini', max_depth = 10, max_features = None, \
    max_leaf_nodes = None, min_samples_leaf = 20, min_samples_split = 2, min_weight_fraction_leaf = 0.0, presort = False,\
         random_state = 100, splitter = 'best')
clf = clf.fit(X, y)



y_pred = clf.predict(X)
print("Tree 1 confusion matrix")
print(metrics.confusion_matrix(y, y_pred))

dot_data = tree.export_graphviz(clf, out_file = 'amazon_tools.dot', feature_names = X.columns)

print("Result for tree 1")
#print("Nationality1, ProvinceResidence1, ModeTransport1, Destination1, FlyingCom & AccessTime")

kf=KFold(n_splits=7, random_state=7, shuffle = True)
result_accuracy=cross_val_score(clf, X, y, scoring = 'accuracy', cv = kf)
print(result_accuracy)
print(np.mean(result_accuracy))



#Kfold for recall and precision
precision_scores = []
recall_scores = []
fscore_scores = []
for train_index, test_index in kf.split(X):
 #   print("TRAIN:", train_index, "TEST:", test_index)
    X_train = X.iloc[train_index]
    X_test =  X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    precision_scores.append(precision_score(y_test, y_pred, average='macro'))
    recall_scores.append(recall_score(y_test, y_pred, average='macro'))
    fscore_scores.append(f1_score(y_test, y_pred, average='macro'))

print(precision_scores)
print("Precision: %0.4f (+/- %0.4f)" % (np.mean(precision_scores), np.std(precision_scores) * 2))
print(recall_scores)
print("Recall: %0.4f (+/- %0.4f)" % (np.mean(recall_scores), np.std(recall_scores) * 2))
print(fscore_scores)
print("Fscore: %0.4f (+/- %0.4f)" % (np.mean(fscore_scores), np.std(fscore_scores) * 2))
dot_data = tree.export_graphviz(clf, out_file = 'airline_final.dot', feature_names = X.columns)

