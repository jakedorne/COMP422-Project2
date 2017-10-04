import numpy as np

def sigmoid(x, deriv=False):
    if(deriv==True):
        return (x*(1-x))

    return 1/(1+np.exp(-x))

#input data
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])

#output data
y = np.array([[0],
             [1],
             [1],
             [0]])

np.random.seed(1)

#synapses
syn0 = 2 * np.random.random((3,4)) - 1  # 2x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
syn1 = 2 * np.random.random((4,1)) - 1  # 4x1 matrix of weights. (4 nodes x 1 output) - no bias term in the hidden layer.

#training step
for j in range(100000):

    # Calculate forward through the network.
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))

    # Back propagation of errors using the chain rule.
    l2_error = y - l2
    if(j % 10000) == 0:   # Only print the error every 10000 steps, to save time and limit the amount of output.
        print("Error: " + str(np.mean(np.abs(l2_error))))

    l2_delta = l2_error * sigmoid(l2, deriv=True)

    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * sigmoid(l1,deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print("Output:")
print(l2)
