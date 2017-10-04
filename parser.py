import sys
import numpy as np
import random

def digits():
    train = []
    test = []
    f = open("digits/digits{}".format(sys.argv[1]), "r")
    text = f.read().split("\n")
    random.shuffle(text)

    inputs = []
    labels = []
    for line in text:
        if line != "":
            inputs.append(line.split()[:-1])
            labels.append(line.split()[-1])

    training_inputs = np.array(inputs[:len(inputs)/2]).astype(np.float)
    training_labels = np.array(labels[:len(labels)/2]).astype(np.float)
    test_inputs = np.array(inputs[len(inputs)/2:]).astype(np.float)
    test_labels = np.array(labels[len(labels)/2:]).astype(np.float)

    return training_inputs, training_labels, test_inputs, test_labels


def uci_data(name):
    train = []
    test = []
    f = open("uci-datasets/{}.data".format(name), "r")
    text = f.read().split("\n")
    random.shuffle(text)

    inputs = []
    labels = []
    for line in text:
        if line != "":
            inputs.append(line.split(',')[:-1])
            labels.append(line.split(',')[-1])

    training_inputs = np.array(inputs[:len(inputs)/2]).astype(np.float)
    training_labels = np.array(labels[:len(labels)/2]).astype(np.float)
    test_inputs = np.array(inputs[len(inputs)/2:]).astype(np.float)
    test_labels = np.array(labels[len(labels)/2:]).astype(np.float)

    return training_inputs, training_labels, test_inputs, test_labels
