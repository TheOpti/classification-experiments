import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from plot_utils import plot_confusion_matrix

if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# import some data to play with
data = datasets.load_iris()
X = data.data
y = data.target
class_names = data.target_names

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01)
y_pred = classifier.fit(X_train, y_train).predict(X_test)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.savefig('./visualizations/example_confusion_matrix.png')
plt.clf()

# # Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Confusion matrix - normalized')
plt.savefig('./visualizations/example_confusion_matrix_normalized.png')
plt.clf()
