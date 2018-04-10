import pandas as pd
import matplotlib.pyplot as plt

iris = pd.read_csv("./data/iris/iris.csv")

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

    ax.legend(plot_types, ["Iris Setosa", "Iris Versicolor", "Iris Virginica"], loc=2)
    figure.savefig("./visualizations/" + type + ".png", bbox_inches='tight')


