import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iris = pd.read_csv("./data/iris/iris.csv")

iris.drop('Id', axis=1, inplace=True)
param_cols = list(iris.columns.values)

matrix = iris[param_cols].values
labels = np.unique(iris['Species'])

sepal_length_setosa, sepal_width_setosa = [], []
sepal_length_versicolor, sepal_width_versicolor = [], []
sepal_length_virginica, sepal_width_virginica = [], []

petal_length_setosa, petal_width_setosa = [], []
petal_length_versicolor, petal_width_versicolor = [], []
petal_length_virginica, petal_width_virginica = [], []

sepal_length_idx = 0
sepal_width_idx = 1
petal_length_idx = 2
petal_width_idx = 3

for n, elem in enumerate(matrix):
    if elem[-1] == 'Iris-setosa':
        sepal_length_setosa.append(matrix[n][sepal_length_idx])
        sepal_width_setosa.append(matrix[n][sepal_width_idx])
        petal_length_setosa.append(matrix[n][petal_length_idx])
        petal_width_setosa.append(matrix[n][petal_width_idx])
    elif elem[-1] == 'Iris-versicolor':
        sepal_length_versicolor.append(matrix[n][sepal_length_idx])
        sepal_width_versicolor.append(matrix[n][sepal_width_idx])
        petal_length_versicolor.append(matrix[n][petal_length_idx])
        petal_width_versicolor.append(matrix[n][petal_width_idx])
    elif elem[-1] == 'Iris-virginica':
        sepal_length_virginica.append(matrix[n][sepal_length_idx])
        sepal_width_virginica.append(matrix[n][sepal_width_idx])
        petal_length_virginica.append(matrix[n][petal_length_idx])
        petal_width_virginica.append(matrix[n][petal_width_idx])

sepal_fig = plt.figure()
ax = sepal_fig.add_subplot(111)

type1 = ax.scatter(sepal_length_setosa, sepal_width_setosa, s=50, c='red')
type2 = ax.scatter(sepal_length_versicolor, sepal_width_versicolor, s=50, c='green')
type3 = ax.scatter(sepal_length_virginica, sepal_width_virginica, s=50, c='blue')

ax.set_title('Sepal size from Iris dataset', fontsize=14)
ax.set_xlabel('Sepal length (cm)')
ax.set_ylabel('Sepal width (cm)')
ax.legend([type1, type2, type3], ["Iris Setosa", "Iris Versicolor", "Iris Virginica"], loc=2)

ax.grid(True, linestyle='-', color='0.75')

sepal_fig.savefig("./visualizations/iris_sepal.png", bbox_inches='tight')

petal_fig = plt.figure()
ax = petal_fig.add_subplot(111)

type1 = ax.scatter(petal_length_setosa, petal_width_setosa, s=50, c='red')
type2 = ax.scatter(petal_length_versicolor, petal_width_versicolor, s=50, c='green')
type3 = ax.scatter(petal_length_virginica, petal_width_virginica, s=50, c='blue')

ax.set_title('Petal size from Iris dataset', fontsize=14)
ax.set_xlabel('Petal length (cm)')
ax.set_ylabel('Petal width (cm)')
ax.legend([type1, type2, type3], ["Iris Setosa", "Iris Versicolor", "Iris Virginica"], loc=2)

ax.grid(True, linestyle='-', color='0.75')

petal_fig.savefig("./visualizations/iris_petal.png", bbox_inches='tight')
