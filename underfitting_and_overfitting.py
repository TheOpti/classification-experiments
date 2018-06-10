print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def true_fun(X):
    return np.cos(2 * np.pi * X)

np.random.seed()

n_samples = 20
degrees = [1, 4, 12]

X = np.arange(0.05, 1, 0.1)
y = true_fun(X) + np.random.randn(X.shape[0]) * 0.1

plt.figure(figsize=(12, 5))
for i in range(len(degrees)):
    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X[:, np.newaxis], y)

    X_test = np.linspace(0, 1, 100)
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]))
    plt.scatter(X, y, edgecolor='b', s=20)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.grid(True)
    plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.show()
