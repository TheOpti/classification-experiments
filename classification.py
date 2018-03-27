from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
clf = tree.DecisionTreeClassifier()

for x in range(0, 10):
    print "Starting iteration no. %d..." % (x + 1)

    clf = tree.DecisionTreeClassifier()

    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

    print "X_train length: ", len(X_train)
    print "X_test length: ", len(X_test)

    clf = clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)

    print "Accuracy score: ", accuracy_score(y_test, predicted)
    print "\n"
