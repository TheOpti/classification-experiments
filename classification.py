import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
number_of_experiments = 100

def train_tree_classifier(number_of_experiments, **options):
    results = []

    for i in range(0, number_of_experiments):
        clf = tree.DecisionTreeClassifier(**options)

        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

        clf = clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        score = accuracy_score(y_test, predicted)

        results.append(score)

    return np.mean( np.array(results) )


result = train_tree_classifier(number_of_experiments, criterion='gini')
print "Criterion = 'gini': ", result

result = train_tree_classifier(number_of_experiments, criterion='entropy')
print "Criterion = 'entropy': ", result

result = train_tree_classifier(number_of_experiments, max_depth=1)
print "max_depth = 1: ", result

result = train_tree_classifier(number_of_experiments, max_depth=2)
print "max_depth = 2: ", result

result = train_tree_classifier(number_of_experiments, max_depth=3)
print "max_depth = 3: ", result
