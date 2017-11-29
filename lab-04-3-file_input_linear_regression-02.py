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

xy = np.loadtxt('kospi01-1000.csv', delimiter=',', dtype=np.float32)
#xy = np.loadtxt('data-01-test-score02.csv', delimiter=',', dtype=np.float32)
#x_data = xy[:, 0:-1]
#y_data = xy[:, [-1]]
'''
numerator = xy - np.min(xy, 0)
denominator = np.max(xy, 0) - np.min(xy, 0)
result =  numerator / denominator

#print("num", numerator, denominator, result)
print("num", numerator)
print("den", denominator)
print("res",result)

xy = MinMaxScaler(xy)
'''
x_data = xy[:, 0:3]
y_data = xy[:, 3:4]

# Make sure the shape and data are OK
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
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

for step in range(1500001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 100000 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

# Ask my score
...
print("Your score will be ", sess.run(
    hypothesis, feed_dict={X: [[0.33195, 0.3321, 0.3303]]}))

print("Other scores will be ", sess.run(hypothesis,
                                        feed_dict={X: [[331.95, 332.1, 330.3]]}))
...

'''
Your score will be  [[ 181.73277283]]
Other scores will be  [[ 145.86265564]
 [ 187.23129272]]

'''
