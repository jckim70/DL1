'''
This script shows how to predict stock prices using a basic RNN
'''
import tensorflow as tf
import numpy as np
import matplotlib
import os

tf.set_random_seed(777)  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


# train Parameters
seq_length = 5
data_dim = 6      # input x
hidden_dim = 6
output_dim = 4
learning_rate = 0.01
iterations = 50000

# Open, High, Low, Volume, Close
#xy = np.loadtxt('KSTOCK-IN-01-10.csv', delimiter=',')
xy = np.loadtxt('KSTOCK-I402-O4-300.csv', delimiter=',')
xy = xy[::-1]  # reverse order (chronically ordered)
#xy = MinMaxScaler(xy)

#x = xy
#y = xy[:, [-1]]  # Close as label

x = xy[:, 1:7]
y = xy[:, 9:13]

# build a dataset
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next close price
    print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

# train/test split
train_size = int(len(dataY) * 0.95)
#train_size = 290
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 4])

# build a LSTM network
#cell = tf.contrib.rnn.BasicLSTMCell(
cell = tf.nn.rnn_cell.BasicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 4])
predictions = tf.placeholder(tf.float32, [None, 4])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
        #if i % 100 == 0:
        print("[step: {}] loss: {}".format(i, step_loss))

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    rmse_val     = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
    print("================================")
    print("RMSE: {}".format(rmse_val))
    print("================================")


    #print(testX, testY, test_predict)
    print("testY===========================")
    print(testY)
    #print(testY, testY[:, 1] - testY[:, 0])
    print("test_predict===========================")
    print(test_predict)
#    print("difference===========================")
#    print(testY - test_predict)

