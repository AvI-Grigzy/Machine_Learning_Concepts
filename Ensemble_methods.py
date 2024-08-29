import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

bagging_model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
bagging_model.fit(X_train, y_train)
bagging_pred = bagging_model.predict(X_test)
bagging_acc = accuracy_score(y_test, bagging_pred)

random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)
rf_pred = random_forest_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

adaboost_model = AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
adaboost_model.fit(X_train, y_train)
adaboost_pred = adaboost_model.predict(X_test)
adaboost_acc = accuracy_score(y_test, adaboost_pred)

voting_model = VotingClassifier(estimators=[
    ('lr', LogisticRegression()),
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('svc', SVC(probability=True))
], voting='soft')
voting_model.fit(X_train, y_train)
voting_pred = voting_model.predict(X_test)
voting_acc = accuracy_score(y_test, voting_pred)

print(f'Bagging Accuracy: {bagging_acc}')
print(f'Random Forest Accuracy: {rf_acc}')
print(f'AdaBoost Accuracy: {adaboost_acc}')
print(f'Voting Classifier Accuracy: {voting_acc}')

models = ['Bagging', 'Random Forest', 'AdaBoost', 'Voting']
accuracies = [bagging_acc, rf_acc, adaboost_acc, voting_acc]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Ensemble Methods')
plt.ylabel('Accuracy')
plt.title('Comparison of Ensemble Methods Accuracies')
plt.ylim(0, 1)
plt.show()
