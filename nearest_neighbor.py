import numpy as np
import parser
from sklearn import neighbors
from sklearn import tree

x_train, y_train, x_test, y_test = parser.digits()

def nearest_neighbor():
    clf = neighbors.KNeighborsClassifier()
    clf.fit(x_train, y_train)
    confidence = clf.score(x_test, y_test)

    print("Neighbors: ", confidence)

nearest_neighbor()
