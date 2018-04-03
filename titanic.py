import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# sns.countplot(x='Embarked', data=train)
# plt.show()
#
# sns.barplot('Sex', 'Survived', data=train, color="aquamarine")
# plt.show()

# print train.isnull().sum()

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

print 'Training trees...'

tree_results = []
for i in range(0, 100):
    clf = DecisionTreeClassifier()

    X_train, X_test, y_train, y_test = train_test_split(train_data_X, train_data_Y)

    clf = clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    score = accuracy_score(y_test, predicted)

    tree_results.append(score)

print 'Training random forests...'

forest_results = []
for i in range(0, 100):
    clf = RandomForestClassifier()

    X_train, X_test, y_train, y_test = train_test_split(train_data_X, train_data_Y)

    clf = clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    score = accuracy_score(y_test, predicted)

    forest_results.append(score)

print 'Training ada boost...'

ada_boost_results = []
for i in range(0, 100):
    clf = AdaBoostClassifier(DecisionTreeClassifier(),algorithm="SAMME",n_estimators=200)

    X_train, X_test, y_train, y_test = train_test_split(train_data_X, train_data_Y)

    clf = clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    score = accuracy_score(y_test, predicted)

    ada_boost_results.append(score)

print "Score [decision tree]: ", np.mean( np.array(tree_results) )
print "Score [random forest]: ", np.mean( np.array(forest_results) )
print "Score [ada boost]: ", np.mean( np.array(ada_boost_results) )
