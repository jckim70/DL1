# Lab 5 Logistic Regression Classifier
#
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

xy = np.loadtxt('data-jck-01.csv', delimiter=',', dtype=np.float32)
#100,110,1000,1100,110
#KS  KE  NS   NE   KTS
x_data = xy[:, 0:-1]    # 0,1,2,3
y_data = xy[:, [-1]]    # 4

#print(x_data.shape, y_data.shape)
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data, len(y_data))

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 4])  #N DATA, IN DATA NO
Y = tf.placeholder(tf.float32, shape=[None, 1])  #N개, OUT DATA 갯수

W = tf.Variable(tf.random_normal([4, 1]), name='IN')  # IN DATA NO
b = tf.Variable(tf.random_normal([1]), name='OUT')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(1001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

'''
0 0.82794
200 0.755181
400 0.726355
600 0.705179
800 0.686631
...
9600 0.492056
9800 0.491396
10000 0.490767

...

 [ 1.]
 [ 1.]
 [ 1.]]
Accuracy:  0.762846
'''
