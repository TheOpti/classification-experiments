import numpy as np

from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
number_of_experiments = 10000

results = []

for i in range(0, number_of_experiments):
    clf_gini = tree.DecisionTreeClassifier(criterion='gini')

    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

    clf_gini = clf_gini.fit(X_train, y_train)
    predicted = clf_gini.predict(X_test)
    score = accuracy_score(y_test, predicted)

    results.append(score)

results_np = np.array(results)
print "Criterion = 'gini': ", np.mean(results_np)

results = []

for i in range(0, number_of_experiments):
    clf = tree.DecisionTreeClassifier(criterion='entropy')

    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

    clf = clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    score = accuracy_score(y_test, predicted)

    results.append(score)

results_np = np.array(results)
print "Criterion = 'entropy': ", np.mean(results_np)

results = []

for i in range(0, number_of_experiments):
    clf = tree.DecisionTreeClassifier(max_depth=1)

    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

    clf = clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    score = accuracy_score(y_test, predicted)

    results.append(score)

results_np = np.array(results)
print "max_depth = 1: ", np.mean(results_np)

results = []

for i in range(0, number_of_experiments):
    clf = tree.DecisionTreeClassifier(max_depth=2)

    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

    clf = clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    score = accuracy_score(y_test, predicted)

    results.append(score)

results_np = np.array(results)
print "max_depth = 2: ", np.mean(results_np)
