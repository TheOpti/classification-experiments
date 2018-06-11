import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

titanic = pd.read_csv("./data/titanic/titanic.csv")

titanic['travel_buds'] = titanic["sibsp"] + titanic["parch"]
titanic['travel_alone'] = np.where(titanic['travel_buds'] > 0, 0, 1)

plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

print 'Embarked'
sns.countplot(x='embarked', data=titanic)
plt.savefig('./visualizations/titanic_embarked.png')
plt.clf()

print 'Survived by sex'
sns.barplot(x='sex', y='survived', data=titanic)
plt.savefig('./visualizations/titanic_survived_by_sex.png')
plt.clf()

print 'Travel alone'
sns.countplot(x='travel_alone', data=titanic)
plt.savefig('./visualizations/titanic_travel_alone.png')
plt.clf()

print 'Class/sex ratio'
sns.factorplot('pclass', hue='sex', data=titanic,  kind='count')
plt.savefig('./visualizations/titanic_class_sex_ratio.png')
plt.clf()

print 'Class/survived ratio'
g = sns.factorplot(x="sex", y="survived", col="pclass",
                   data=titanic, saturation=.5,
                   kind="bar", ci=None, aspect=.6)
g.set_axis_labels("", "Survival Rate")\
    .set_xticklabels(["Women", "Men"])\
    .set_titles("{col_name} {col_var}")\
    .set(ylim=(0, 1))\
    .despine(left=True)
plt.savefig('./visualizations/titanic_class_survived_ratio.png')
plt.clf()
