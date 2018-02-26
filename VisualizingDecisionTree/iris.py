import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
print iris.feature_names
print iris.target_names
print iris.data[0]
print iris.target[0]
for i in range(len(iris.target)):
    print "Example %d: label %s,features %s" %(i, iris.target[i], iris.data[i])
test_idx = [0,50,100]

#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx,axis=0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print test_target

print clf.predict(test_data)

#viz code
from sklearn.externals.six import StringIO
import graphviz
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris")

print test_data[1],test_target[1]
print iris.feature_names, iris.target_names

















