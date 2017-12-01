# Lab 4 Multi-variable linear regression
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
#    return numerator / (denominator + 1e-7)
    return numerator / (denominator)

xy = np.loadtxt('KSTOCK-I222-O2-290.csv', delimiter=',')
xy = xy[::-1]  # reverse order (chronically ordered)
x_data = xy[:, 1:7]
y_data = xy[:, 7:9]

# Make sure the shape and data are OK
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

testxy = np.loadtxt('KSTOCK-I222-O2-010.csv', delimiter=',')
testxy = testxy[::-1]  # reverse order (chronically ordered)
testx_data = testxy[:, 1:7]
testy_data = testxy[:, 7:9]


# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 6])
Y = tf.placeholder(tf.float32, shape=[None, 2])

W = tf.Variable(tf.random_normal([6, 2]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')


# Hypothesis
hypothesis = tf.matmul(X, W)

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(1800001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 100000 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

# Ask my score
...
print(testx_data)
print(testy_data)
print("Next day ", sess.run(hypothesis,
                            #feed_dict={X: [[0.336,0.3375,0.3349,0.3359,0.2595]]}
                     feed_dict={X: testx_data}
                            )
      )

#test_predict = sess.run(Y_pred, feed_dict={X: testX})

# 0.33625	0.3348
