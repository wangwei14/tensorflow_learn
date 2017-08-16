# coding=utf-8

import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data

mnist = tensorflow.examples.tutorials.mnist.input_data.read_data_sets("MNIST_data/", one_hot=True)

# 权重初始化
def weight_variable(shape):
	# stddev -> Standard Diviation
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

# 卷积和池化
def conv2d(x, W):
	# input -> tensor of shape [batch, in_height, in_width, in_channels]
	# filter -> tensor of shape [filter_height, filter_width, in_channels, out_channels]
	# strides -> The stride of the sliding window for each dimension of input.
	# output -> default the same size of input
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	# ksize -> The size of the window for each dimension of the input tensor.
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

sess = tf.InteractiveSession()

# 第一层卷积由一个卷积接一个max pooling完成.
# 卷积在每个5x5的patch中算出32个特征.前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目.
# 而对于每一个输出通道都有一个对应的偏置量.
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# 为了用这一层，我们把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数.
# -1 -> flatten or infer the shape
x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 密集连接层
# 现在，图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片.
# 我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 *64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
# 代表一个神经元的输出在dropout中保持不变的概率.这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout.
# TensorFlow除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的scale.所以用dropout的时候可以不用考虑scale.
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Softmax
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 训练和评估模型
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.global_variables_initializer())
for i in range(2000):
	batch = mnist.train.next_batch(50)
	if i % 100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
		print "step %d, training accuracy %g" % (i, train_accuracy)
	train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
	# print tf.shape(x_image).eval(feed_dict={x: batch[0]})
	# print tf.shape(h_conv1).eval(feed_dict={x: batch[0]})
	# print tf.shape(h_pool1).eval(feed_dict={x: batch[0]})
	# print tf.shape(h_conv2).eval(feed_dict={x: batch[0]})
	# print tf.shape(h_pool2).eval(feed_dict={x: batch[0]})
	# print tf.shape(h_pool2_flat).eval(feed_dict={x: batch[0]})
	# print tf.shape(h_fc1).eval(feed_dict={x: batch[0]})
	# print tf.shape(h_fc1_drop).eval(feed_dict={x: batch[0], keep_prob: 0.2})

print "test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

sess.close()

