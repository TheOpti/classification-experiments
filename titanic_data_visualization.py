import pandas as pd
import matplotlib.pyplot as plt # data visualization
import seaborn as sns

train = pd.read_csv("./data/titanic/titanic.csv")

plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

sns.countplot(x='embarked', data=train)
plt.show()

sns.barplot(x='sex', y='survived', data=train)
plt.show()