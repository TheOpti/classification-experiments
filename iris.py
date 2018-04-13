import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

print 'Load data'
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

tree_param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': np.arange(2, 8),
}

tree = GridSearchCV(DecisionTreeClassifier(), tree_param_grid)

tree.fit(X_train, y_train)
tree_preds = tree.predict(X_test)
tree_performance = accuracy_score(y_test, tree_preds)

print '\nDecision tree:'
print 'Best params: ', tree.best_params_
print 'Best scores: ', tree.best_score_
print 'My score: ', tree_performance

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

print '\nRandom forest:'
print 'Best params: ', random_forest.best_params_
print 'Best scores: ', random_forest.best_score_
print 'My score: ', random_forest_performance

ada_boost_param_grid = {
    'base_estimator': [DecisionTreeClassifier()],
    'n_estimators': np.arange(30, 200, 4)
}

ada_boost = GridSearchCV(AdaBoostClassifier(), ada_boost_param_grid)

ada_boost.fit(X_train, y_train)
ada_boost_preds = ada_boost.predict(X_test)
ada_boost_performance = accuracy_score(y_test, ada_boost_preds)

print '\nAda Boost:'
print 'Best params: ', ada_boost.best_params_
print 'Best scores: ', ada_boost.best_score_
print 'My score: ', ada_boost_performance

gradient_boost_param_grid = {
    'loss': ['deviance'],
    'n_estimators': np.arange(50, 200, 4),
    'criterion': ['friedman_mse', 'mse', 'mae']
}

gradient_boost = GridSearchCV(GradientBoostingClassifier(), gradient_boost_param_grid)

gradient_boost.fit(X_train, y_train)
gradient_preds = gradient_boost.predict(X_test)
gradient_performance = accuracy_score(y_test, gradient_preds)

print '\nGradientBoostingClassifier:'
print 'Best params: ', gradient_boost.best_params_
print 'Best scores: ', gradient_boost.best_score_
print 'My score: ', gradient_performance
