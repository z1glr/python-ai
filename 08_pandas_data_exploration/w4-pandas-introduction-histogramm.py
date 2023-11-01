
import pandas as pd
import matplotlib.pyplot as plt
# import prettyprinter as pp # unused

# import iris dataset for test of AI methods into pandas dataframe
iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# get basic information about the data set
print(iris.info()) # new

# print the shape of the data
print(iris.shape) # new

# print the first 5 rows of the data
print(iris.head())

# print the last rows of the data
print(iris.tail())

# print what columns exist
print(iris.columns)

# name columns
iris.columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
print(iris.columns)

# now we can use the more descriptive column names for robust code
# print the class distribution
print(iris.groupby('class').size()) # new

# get some overall statistics
print(iris.describe()) # new

# or get some statistics for each class
print(iris.groupby('class').describe())

# plot features of iris data
iris.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# plot histogram of iris data
iris.hist()
plt.show()

# plot class dependent box-and whisker plots
iris.boxplot('sepal-length', 'class') # new
plt.show()
