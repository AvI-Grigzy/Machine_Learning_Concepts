import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

adaboost_model = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost_model.fit(X_train, y_train)
adaboost_pred = adaboost_model.predict(X_test)
adaboost_acc = accuracy_score(y_test, adaboost_pred)

gradient_boosting_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
gradient_boosting_model.fit(X_train, y_train)
gradient_boosting_pred = gradient_boosting_model.predict(X_test)
gradient_boosting_acc = accuracy_score(y_test, gradient_boosting_pred)

xgboost_model = XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='logloss', random_state=42)
xgboost_model.fit(X_train, y_train)
xgboost_pred = xgboost_model.predict(X_test)
xgboost_acc = accuracy_score(y_test, xgboost_pred)

print(f'AdaBoost Accuracy: {adaboost_acc}')
print(f'Gradient Boosting Accuracy: {gradient_boosting_acc}')
print(f'XGBoost Accuracy: {xgboost_acc}')

models = ['AdaBoost', 'Gradient Boosting', 'XGBoost']
accuracies = [adaboost_acc, gradient_boosting_acc, xgboost_acc]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Boosting Methods Comparison')
plt.ylim(0.8, 1.0)
plt.show()
