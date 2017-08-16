# coding=utf-8

import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data

mnist = tensorflow.examples.tutorials.mnist.input_data.read_data_sets("MNIST_data/", one_hot=True)
# print mnist.train.next_batch(10)[0][1]

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

# 非常漂亮的成本函数是“交叉熵”（cross-entropy）
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	
	for i in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})

	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels})
