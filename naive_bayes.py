import numpy as np
import parser
import sys
from sklearn.naive_bayes import GaussianNB

x_train, y_train, x_test, y_test = parser.uci_data(sys.argv[1])

def naive_bayes():
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    confidence = clf.score(x_test, y_test)

    print("Accuracy: ", confidence)

naive_bayes()
