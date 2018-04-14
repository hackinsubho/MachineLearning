from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data

y = iris.target

#print(X.shape)
#print(y.shape)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
#print(knn)

knn.fit(X,y)

X_new = [[3,5,4,2],[5,4,3,2]]


print(knn.predict(X_new))

knn = KNeighborsClassifier(n_neighbors=5)
#print(knn)

knn.fit(X,y)

X_new = [[3,5,4,2],[5,4,3,2]]


print(knn.predict(X_new))


"""using a different classification model"""

from sklearn.linear_model  import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X,y)
print(logreg.predict(X_new))