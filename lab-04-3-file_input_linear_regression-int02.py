# Lab 4 Multi-variable linear regression
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

#xy = np.loadtxt('data-01-test-score02.csv', delimiter=',', dtype=np.float32)
xy = np.loadtxt('KOSPI01.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:3]
y_data = xy[:, 3:5]

# Make sure the shape and data are OK
print(x_data.shape, x_data)
print(y_data.shape, y_data)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 2])

W = tf.Variable(tf.random_normal([3, 2]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.sparse_matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
#cost = tf.reduce_max(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(5001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 1000 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

# Ask my score
print(" P1 ", sess.run(hypothesis, feed_dict={X: [[73, 80, 75]]}))

