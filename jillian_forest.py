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
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


data = pd.read_csv("master.csv", sep=',', header = 0)

data['tools_used_not']=data['tools_used_not'].astype('category').cat.codes

data['Customer_size']=data['Customer_size'].astype('category').cat.codes

data['Geo_Code']=data['Geo_Code'].astype('category').cat.codes

## Forest 1 - min_samples_leaf = 1

y = data['tools_used_not']
X = data[['Customer_size', 'Geo_Code', 'ProductsUsed', 'totalBilled', 'customer_age']] 


X, y = make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)
clf = RandomForestClassifier(n_estimators=500, max_depth=2,
                              random_state=0)
clf.fit(X, y)  
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=2, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False)

print(clf.feature_importances_)

print(clf.predict([[0, 0, 0, 0]]))


# Forest 2 - min_samples_leaf = 50 - n_estimators = 1000


y = data['tools_used_not']
X = data[['Customer_size', 'Geo_Code', 'ProductsUsed', 'totalBilled', 'customer_age']] 


X, y = make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)
clf = RandomForestClassifier(n_estimators=1000, max_depth=2,
                              random_state=0)
clf.fit(X, y)  
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=2, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=50, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False)

print(clf.feature_importances_)

print(clf.predict([[0, 0, 0, 0]]))

# Forest 3 - min_samples_leaf = 50 - n_estimators = 5000


y = data['tools_used_not']
X = data[['Customer_size', 'Geo_Code', 'ProductsUsed', 'totalBilled', 'customer_age']] 


X, y = make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)
clf = RandomForestClassifier(n_estimators=5000, max_depth=2,
                              random_state=0)
clf.fit(X, y)  
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=2, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=50, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=5000, n_jobs=-1,
            oob_score=True, random_state=0, verbose=0, warm_start=False)

print(clf.feature_importances_)

print(clf.predict([[0, 0, 0, 0]]))


