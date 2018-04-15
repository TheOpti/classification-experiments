import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

print 'Load data'
train = pd.read_csv("./data/titanic/titanic.csv")
train_data = train.copy()

age_median = train_data["age"].median(skipna=True)
fare_median = train_data["fare"].median(skipna=True)

print 'Prepare data'
train_data["age"].fillna(age_median, inplace=True)
train_data["fare"].fillna(fare_median, inplace=True)
train_data["sex"].fillna('male', inplace=True)
train_data["sibsp"].fillna(0, inplace=True)
train_data["parch"].fillna(0, inplace=True)
train_data["embarked"].fillna("S", inplace=True)

train_data['travel_buds'] = train_data["sibsp"] + train_data["parch"]
train_data['travel_alone'] = np.where(train_data['travel_buds'] > 0, 0, 1)

train_data.drop('cabin', axis=1, inplace=True)
train_data.drop('ticket', axis=1, inplace=True)
train_data.drop('name', axis=1, inplace=True)
train_data.drop('home.dest', axis=1, inplace=True)
train_data.drop('boat', axis=1, inplace=True)
train_data.drop('body', axis=1, inplace=True)
train_data = train_data.dropna(how='any', axis=0)

train_data['sex'] = train_data['sex'].map({'male': 1, 'female': 0})

# Create categorical data for Pclass, Embarked
train_data_2 = pd.get_dummies(train_data, columns=["pclass"])
final_train_data = pd.get_dummies(train_data_2, columns=["embarked"])

train_data_X = final_train_data[final_train_data.columns.difference(['survived'])]
train_data_Y = final_train_data['survived']

X_train, X_test, y_train, y_test = train_test_split(train_data_X, train_data_Y)

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
