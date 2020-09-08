# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix

from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# DATA SUMMARY
# load the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv("iris.csv", names=names)

# shape of the dataset
# instances = rows
# attributes = columns
print("\nSHAPE-(Rows, Columns): " + str(dataset.shape))

# head (peek at the data)
print("\nHEAD:\n " + str(dataset.head(20)))

# descriptions
print("\nDESCRIPTIONS:")
print(dataset.describe())

# class distribution
print("\nCLASS DISTRIBUTION:")
print(dataset.groupby('class').size())

# DATA VISUALIZATION
# Univariate plots
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(
    2, 2), sharex=False, sharey=False)
pyplot.show()

# histogram
dataset.hist()
pyplot.show()

# Multivariate plots
# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

# CREATE VALIDATION DATASET
# hold back 20% of data for validating trained models
array = dataset.values
x = array[:, 0:4]
y = array[:, 4]
x_train, x_validation, y_train, y_validation = train_test_split(
    x, y, test_size=0.20, random_state=1)

# use 10-fold cross validation to estimate model accuracy

# Spot check algorithms
models = []

# Logistic regression
models.append(('LR', LogisticRegression(
    solver='liblinear', multi_class='ovr')))
# Linear Discriminant Analysis
models.append(('LDA', LinearDiscriminantAnalysis()))
# K-Nearest Neighbors
models.append(('KNN', KNeighborsClassifier()))
# Classification and regression trees
models.append(('CART', DecisionTreeClassifier()))
# Gaussian Naive Bayes
models.append(('NB', GaussianNB()))
# Support Vector Machines
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(
        model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# Compare algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Data suggests SVM is most accurate model
# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(x_train, y_train)
predictions = model.predict(x_validation)

# Evaluate predictions
print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))
