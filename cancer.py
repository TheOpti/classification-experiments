import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from plot_utils import plot_learning_curve, plot_validation_curve, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

print 'Load data'
train = pd.read_csv("./data/cancer/cancer.csv")
train_data = train.copy()

train_data['diagnosis'] = train_data['diagnosis'].map({'M': 1, 'B': 0})
train_data.drop('id', axis=1, inplace=True)

train_data_X = train_data[train_data.columns.difference(['diagnosis'])]
train_data_Y = train_data['diagnosis']
class_names = ['Malignant', 'Benign']

X_train, X_test, y_train, y_test = train_test_split(train_data_X, train_data_Y)

tree_param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': np.arange(2, 8),
    'min_samples_split': np.arange(2, 5),
    'min_samples_leaf': [1, 2, 3]
}

tree = GridSearchCV(DecisionTreeClassifier(), tree_param_grid)
tree.fit(X_train, y_train)

best = tree.best_estimator_
best.fit(X_train, y_train)
best_preds = best.predict(X_test)
best_performance = accuracy_score(y_test, best_preds)

print '\nDecision tree:'
print 'Best params: ', tree.best_params_
print 'Best score: ', tree.best_score_
print 'My best score: ', best_performance

print 'Classification report:'
print(classification_report(y_test, best_preds, target_names=class_names))

print 'Confusion matrix plotting...'
cnf_matrix = confusion_matrix(y_test, best_preds)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.savefig('./visualizations/cancer_tree_confusion_matrix.png')
plt.clf()

for param in tree_param_grid:
    print 'Validation curve for ' + param + ' plotting...'
    title = 'Decision tree - Validation curve (' + param + ')'
    plot_validation_curve(best, title, train_data_X, train_data_Y, param, tree_param_grid[param])
    plt.savefig('./visualizations/cancer_tree_validation_' + param + '.png')
    plt.clf()

print 'Learning curve plotting...'
title = 'Decision tree - Learning curve'
plot_learning_curve(tree, title, train_data_X, train_data_Y,)
plt.savefig('./visualizations/cancer_tree_learning_curve.png')
plt.clf()


forest_param_grid = {
    'n_estimators': np.arange(10, 26, 2),
    'criterion': ['gini', 'entropy'],
    'max_depth': np.arange(2, 8),
    'bootstrap': [True, False]
}

random_forest = GridSearchCV(RandomForestClassifier(), forest_param_grid)
random_forest.fit(X_train, y_train)

best = random_forest.best_estimator_
best.fit(X_train, y_train)
best_preds = best.predict(X_test)
best_performance = accuracy_score(y_test, best_preds)

print '\nRandom forest:'
print 'Best params: ', random_forest.best_params_
print 'Best score: ', random_forest.best_score_
print 'My best core: ', best_performance

print 'Classification report:'
print(classification_report(y_test, best_preds, target_names=class_names))

print 'Confusion matrix plotting...'
cnf_matrix = confusion_matrix(y_test, best_preds)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.savefig('./visualizations/cancer_forest_confusion_matrix.png')
plt.clf()

for param in forest_param_grid:
    print 'Validation curve for ' + param + ' plotting...'
    title = 'Random forest - Validation curve (' + param + ')'
    plot_validation_curve(best, title, train_data_X, train_data_Y, param, forest_param_grid[param])
    plt.savefig('./visualizations/cancer_forest_validation_' + param + '.png')
    plt.clf()

print 'Learning curve plotting...'
title = 'Random forest - Learning curve'
plot_learning_curve(random_forest, title, train_data_X, train_data_Y)
plt.savefig('./visualizations/cancer_forest_learning.png')
plt.clf()


ada_boost_param_grid = {
    'n_estimators': np.arange(12, 52, 4),
    'learning_rate': [1, 1.5, 2.0, 2.5, 3.0]
}

ada_boost = GridSearchCV(AdaBoostClassifier(), ada_boost_param_grid)
ada_boost.fit(X_train, y_train)

best = ada_boost.best_estimator_
best.fit(X_train, y_train)
best_preds = best.predict(X_test)
best_performance = accuracy_score(y_test, best_preds)

print '\nAda Boost:'
print 'Best params: ', ada_boost.best_params_
print 'Best score: ', ada_boost.best_score_
print 'My best core: ', best_performance

print 'Classification report:'
print(classification_report(y_test, best_preds, target_names=class_names))

print 'Confusion matrix plotting...'
cnf_matrix = confusion_matrix(y_test, best_preds)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.savefig('./visualizations/cancer_ada_boost_confusion_matrix.png')
plt.clf()

for param in ada_boost_param_grid:
    print 'Validation curve for ' + param + ' plotting...'
    title = 'AdaBoost - Validation curve (' + param + ')'
    plot_validation_curve(best, title, train_data_X, train_data_Y, param, ada_boost_param_grid[param])
    plt.savefig('./visualizations/cancer_ada_boost_validation_' + param + '.png')
    plt.clf()

print 'Learning curve plotting...'
title = 'Ada Boost - Learning Curve'
plot_learning_curve(ada_boost, title, train_data_X, train_data_Y)
plt.savefig('./visualizations/cancer_ada_boost_learning_curve.png')