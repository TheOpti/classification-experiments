import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

print 'Decision tree'

tree_param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': np.arange(2, 8),
}

tree = GridSearchCV(DecisionTreeClassifier(), tree_param_grid)

tree.fit(X_train, y_train)
tree_preds = tree.predict(X_test)
tree_performance = accuracy_score(y_test, tree_preds)

print 'Best params: ', tree.best_params_
print 'Best scores: ', tree.best_score_
print 'My score: ', tree_performance

print 'Random forest'

forest_param_grid = {
    'n_estimators': np.arange(10, 26, 2),
    'criterion': ['gini', 'entropy'],
    'max_depth': np.arange(2, 8),
    'bootstrap': [True, False]
}

random_forest = GridSearchCV(RandomForestClassifier(), forest_param_grid)

random_forest.fit(X_train, y_train)
random_forest_preds = random_forest.predict(X_test)
random_forest_performance = accuracy_score(y_test, random_forest_preds)

print 'Best params: ', random_forest.best_params_
print 'Best scores: ', random_forest.best_score_
print 'My score: ', random_forest_performance

