import tensorflow as tf

# Section 1
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

x.initializer.run()

sub = tf.sub(x, a)
print sub.eval()

sess.close()

# Section 2
state = tf.Variable(0, name = 'counter')
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init_op = tf.initialize_all_variables()

# Section 3
with tf.Session() as sess:
	sess.run(init_op)
	print sess.run(state)
	for _ in range(3):
		sess.run(update)
		print sess.run(state)

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.mul(intermed, input1)

with tf.Session() as sess:
	result = sess.run([mul, intermed])
	print result

# Section 4
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
	print sess.run([output], feed_dict = {input1:[7.], input2:[2.]})
