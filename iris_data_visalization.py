import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iris = pd.read_csv("./data/iris/iris.csv")

iris.drop('Id', axis=1, inplace=True)
param_cols = list(iris.columns.values)

matrix = iris[param_cols].values
labels = np.unique(iris['Species'])

xcord1, ycord1 = [], []
xcord2, ycord2 = [], []
xcord3, ycord3 = [], []

petal_length_idx = 2
petal_width_idx = 3

for n, elem in enumerate(matrix):
    if elem[-1] == 'Iris-setosa':
        xcord1.append(matrix[n][petal_length_idx])
        ycord1.append(matrix[n][petal_width_idx])
    elif elem[-1] == 'Iris-versicolor':
        xcord2.append(matrix[n][petal_length_idx])
        ycord2.append(matrix[n][petal_width_idx])
    elif elem[-1] == 'Iris-virginica':
        xcord3.append(matrix[n][petal_length_idx])
        ycord3.append(matrix[n][petal_width_idx])

fig = plt.figure()
ax = fig.add_subplot(111)

type1 = ax.scatter(xcord1, ycord1, s=50, c='red')
type2 = ax.scatter(xcord2, ycord2, s=50, c='green')
type3 = ax.scatter(xcord3, ycord3, s=50, c='blue')

ax.set_title('Petal size from Iris dataset', fontsize=14)
ax.set_xlabel('Petal length (cm)')
ax.set_ylabel('Petal width (cm)')
ax.legend([type1, type2, type3], ["Iris Setosa", "Iris Versicolor", "Iris Virginica"], loc=2)

ax.grid(True, linestyle='-', color='0.75')

fig.savefig("./visualizations/iris_petal_width_height.png", bbox_inches='tight')



