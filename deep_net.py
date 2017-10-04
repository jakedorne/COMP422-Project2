import sys
import tensorflow as tf
import parser
import numpy as np
import parser
import random
import progressbar


training_inputs, training_labels, test_inputs, test_labels = parser.uci_data(sys.argv[1])

n_features = len(training_inputs[0])
n_classes = len(test_labels[0])

batch_size = 100

hl1_nodes = n_features
hl2_nodes = n_features
hl3_nodes = n_features

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)


hl1 = {'weights': tf.Variable(tf.random_normal([n_features, hl1_nodes])),
        'biases': tf.Variable(tf.random_normal([hl1_nodes]))}

hl2 = {'weights': tf.Variable(tf.random_normal([hl1_nodes, hl2_nodes])),
        'biases': tf.Variable(tf.random_normal([hl2_nodes]))}

hl3 = {'weights': tf.Variable(tf.random_normal([hl2_nodes, hl3_nodes])),
        'biases': tf.Variable(tf.random_normal([hl3_nodes]))}

output_layer = {'weights': tf.Variable(tf.random_normal([hl3_nodes, n_classes])),
        'biases': tf.Variable(tf.random_normal([n_classes]))}

saver = tf.train.Saver()

def neural_network_model(data):
    l1 = tf.add(tf.matmul(data, hl1['weights']), hl1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hl2['weights']), hl2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hl3['weights']), hl3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    epochs = 100

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            epoch_loss = 0
            i = 0
            while i < len(training_inputs):
                start = i
                end = i + batch_size

                batch_x = np.array(training_inputs[start:end])
                batch_y = np.array(training_labels[start:end])

                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})

                epoch_loss += c

                i += batch_size

            print("Epoch", epoch+1, "completed out of", epochs, "loss:", epoch_loss)
            saver.save(sess,"models/{} {}/model-{}.ckpt".format(sys.argv[1], sys.argv[2], sys.argv[1]))

            if(epoch_loss == 0):
                break

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print("Accuracy: ", accuracy.eval({x: test_inputs, y: test_labels}))


train_neural_network(x)
