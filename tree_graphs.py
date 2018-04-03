import graphviz
from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()
clf = tree.DecisionTreeClassifier()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)
clf = clf.fit(X_train, y_train)

dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("iris")
