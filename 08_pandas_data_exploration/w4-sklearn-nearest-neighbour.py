# load iris data from sklean into pandas dataframe
from sklearn.datasets import load_iris
data = load_iris() # new

# split data into test and train sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42) # new

# evaluate k nearest neighbors on data for different nr of neighbors
for ii in range(3, 10, 2):
    print('Anzahl Nachbarn: ', ii)

    # create a k nearest neighbors classifier
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=ii)

    # stuff data into algorithm
    knn.fit(X_train, y_train)

    # now calculate the predictions for the test data
    y_predicted = knn.predict(X_test)

    # print accuracy - that is by default the fraction of correctly classified samples
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, y_predicted)) # new

    # print confusion matrix - the lines are the true classes, the columns are the predicted classes
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test, y_predicted)) # new
    print('')
