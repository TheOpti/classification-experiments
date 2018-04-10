import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = pd.read_csv('./data/iris/iris.csv')

print iris['Species'].value_counts()

iris.drop('Id', axis=1, inplace=True)
grouped_by_species = iris.groupby('Species')

data = {}

for key, item in grouped_by_species:
    data[key] = {
        'Sepal': {
            'SepalLengthCm': item['SepalLengthCm'],
            'SepalWidthCm': item['SepalWidthCm']
        },
        'Petal': {
            'PetalLengthCm': item['PetalLengthCm'],
            'PetalWidthCm': item['PetalWidthCm']
        }
    }

types = ['Petal', 'Sepal']
colors = ['red', 'green', 'blue']

for type in types:
    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.grid(True, linestyle='-', color='0.75')
    plot_types = []

    for idx, species in enumerate(data):
        plot_types.append(ax.scatter(
            data[species][type][type + 'LengthCm'],
            data[species][type][type + 'WidthCm'],
            s=50,
            c=colors[idx])
        )

        ax.set_title(species + ' size from Iris dataset', fontsize=14)
        ax.set_xlabel(species + ' length (cm)')
        ax.set_ylabel(species + 'width (cm)')

    ax.legend(plot_types, ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica'], loc=2)
    figure.savefig('./visualizations/' + type + '_all' + '.png', bbox_inches='tight')

sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)
plt.clf()

print 'Petal length boxplot'
sns.boxplot(x='Species', y='PetalLengthCm', data=iris)
plt.savefig('./visualizations/petal_length.png')
plt.clf()

print 'Petal width boxplot'
sns.boxplot(x='Species', y='PetalWidthCm', data=iris)
plt.savefig('./visualizations/petal_width.png')
plt.clf()

print 'Sepal length boxplot'
sns.boxplot(x='Species', y='SepalLengthCm', data=iris)
plt.savefig('./visualizations/sepal_length.png')
plt.clf()

print 'Sepal width boxplot'
sns.boxplot(x='Species', y='SepalWidthCm', data=iris)
plt.savefig('./visualizations/sepal_width.png')
plt.clf()
