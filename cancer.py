import pandas as pd

print 'Load data'
train = pd.read_csv("./data/cancer/cancer.csv")
train_data = train.copy()

print train.isnull().sum()

