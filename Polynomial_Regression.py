import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81])

degree = 2
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

y_poly_pred = model.predict(X_poly)

plt.scatter(X, y, color='blue', label='Original data')
plt.plot(X, y_poly_pred, color='red', label='Polynomial regression fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Polynomial Regression')
plt.show()

print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')
