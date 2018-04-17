import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

np.random.seed(1000)
nb_classifications = 100

datasets = {
    'wine': load_wine(),
    'digits': load_digits()
}

for key in datasets:
    print '\nData: ', key
    accuracy = []

    for i in range(1, nb_classifications):
        print '\rIteration number: ', i + 1,
        rf = RandomForestClassifier(n_estimators = i)

        acc = cross_val_score(rf, datasets[key].data, datasets[key].target, scoring='accuracy', cv=10).mean()

        accuracy.append(acc)

    # Show results
    plt.figure()
    plt.xlabel('Number of trees')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.plot(accuracy)
    plt.savefig('./visualizations/random_forest_error_' + key + '.png')
    plt.clf()

