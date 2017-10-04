import numpy as np
import parser
import sys
from sklearn import tree

x_train, y_train, x_test, y_test = parser.uci_data(sys.argv[1])

def decision_tree():
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    confidence = clf.score(x_test, y_test)

    print("Accuracy: ", confidence)

decision_tree()
