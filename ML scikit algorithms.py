        
import pandas as pd

dataset = pd.read_csv('Admission.csv')
x = dataset.iloc[:, 1:8]
y = dataset.iloc[:, -1]  


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)


# 1.Random Forest 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state = 0)
classifier.fit(X_train, y_train)

y_pred_randomForest = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix ,accuracy_score
cm_randomForest = confusion_matrix(y_test, y_pred_randomForest)
ac_randomForest = accuracy_score(y_test, y_pred_randomForest)



# 2.Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state = 0)
classifier.fit(X_train, y_train)

y_pred_decisionTree = classifier.predict(X_test)
cm_decisionTree = confusion_matrix(y_test, y_pred_decisionTree)
ac_decisionTree = accuracy_score(y_test, y_pred_decisionTree)



# 3.Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

y_pred_logisticRegression = classifier.predict(X_test)
cm_logisticRegression = confusion_matrix(y_test, y_pred_logisticRegression)
ac_logisticRegression = accuracy_score(y_test, y_pred_logisticRegression)



# 4.K-Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

y_pred_KNN = classifier.predict(X_test)
cm__KNN = confusion_matrix(y_test, y_pred_KNN)
ac__KNN = accuracy_score(y_test, y_pred_KNN)



