import pandas as pd
from sklearn.model_selection  import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("master.csv", sep=',', header=0)


data['tools_used_not']=data['tools_used_not'].astype('category').cat.codes

data['Customer_size']=data['Customer_size'].astype('category').cat.codes

data['Geo_Code']=data['Geo_Code'].astype('category').cat.codes


y=data['tools_used_not']

X = data[['Customer_size', 'Geo_Code', 'ProductsUsed', 'totalBilled', 'customer_age']] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=109)  

scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 

## knn classification - 5
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train, y_train)  

## predict test set 
y_pred = classifier.predict(X_test) 
print(metrics.confusion_matrix(y_test, y_pred)) 


## knn classification - 2
classifier = KNeighborsClassifier(n_neighbors=2)  
classifier.fit(X_train, y_train)  

## predict test set 
y_pred = classifier.predict(X_test) 
print(metrics.confusion_matrix(y_test, y_pred)) 


## knn classification - 10
classifier = KNeighborsClassifier(n_neighbors=10)  
classifier.fit(X_train, y_train)  

## predict test set 
y_pred = classifier.predict(X_test) 
print(metrics.confusion_matrix(y_test, y_pred)) 


## knn classification - 20
classifier = KNeighborsClassifier(n_neighbors=20)  
classifier.fit(X_train, y_train)  

## predict test set 
y_pred = classifier.predict(X_test) 
print(metrics.confusion_matrix(y_test, y_pred)) 

## knn classification - 40
classifier = KNeighborsClassifier(n_neighbors=40)  
classifier.fit(X_train, y_train)  

## predict test set 
y_pred = classifier.predict(X_test) 
print(metrics.confusion_matrix(y_test, y_pred)) 

## knn classification - 100
classifier = KNeighborsClassifier(n_neighbors=100)  
classifier.fit(X_train, y_train)  

## predict test set 
y_pred = classifier.predict(X_test) 
print(metrics.confusion_matrix(y_test, y_pred)) 

## knn classification - 200
classifier = KNeighborsClassifier(n_neighbors=200)  
classifier.fit(X_train, y_train)  

## predict test set 
y_pred = classifier.predict(X_test) 
print(metrics.confusion_matrix(y_test, y_pred))