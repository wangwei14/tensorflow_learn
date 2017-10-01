from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

# ===============================  Data  ======================================


def data_input(filename):
    f = open(filename, 'r')
    ans = []

    data = f.read()
    for sentence in data.split('\n')[:-1]:
        temp = map(int, sentence.split(' ')[:-1])
        ans.append(temp)

    f.close()
    return ans


# generate one-hot representation of words to form sentences
# word embeddings are in the first fully connected layer during training
def next_batch(data, vocabulary_size):
    idx = np.random.randint(len(data))
    batch_tokens = np.zeros((sentence_len, vocabulary_size))
    batch_tokens[np.arange(sentence_len), data[idx]] = 1

    return batch_tokens


input_filename = 'data_process/preprocessed_ptb_data.tsv'
all_data = data_input(input_filename)

# =============================== Hyperparameters ======================================

# Language model parameters
sentence_len = 79
vocab_size = 6051
embedding_size = 80
n_hidden = 128

# Network parameters
stddev_value = 1
learning_rate = 0.01
training_iters = 100000

# ============================== Recurrent ====================================


def RNN(tokens):
    """
    inputs:  tokens:     one-hot representation of input tokens/words
                         [sentence_len, vocab_size]
    outputs: w_seq:      sequence of words at each time step, also one-hot
                         [sentence_len, vocab_size]
    """
    local_len = int(tokens.shape[0])
    tokens_stat = tokens[0:1, :]

    # Fully connected layer
    embeddings = tf.add(tf.matmul(tokens, weights['we']), biases['we'])
    embeddings = tf.reshape(embeddings, [1, local_len, embedding_size])

    # Current data input shape: [sentence_len, embedding_size]
    # Required shape: 'n_steps(sentence_len)' tensors list of shape [embedding_size]
    x = tf.unstack(embeddings, local_len, 1)

    # Define a Socially Conditioned LSTM Cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden)

    # Get LSTM Cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    outputs = tf.reshape(tf.stack(outputs), [local_len, n_hidden])

    # Summarize output tokens from LSTM Cell output
    # outputs shape = [sentence_len, n_hidden]
    w_seq = tf.add(tf.matmul(outputs, weights['ho']), biases['ho'])
    w_seq_stat = w_seq[:-1, :]
    w_seq = tf.concat([tokens_stat, w_seq_stat], 0)

    return w_seq

# =============================== Training ====================================


# Graph inputs
init_tokens = tf.placeholder("float", [sentence_len, vocab_size])

# Weights and biases
weights = {
    'we': tf.Variable(tf.random_normal([vocab_size, embedding_size], stddev=stddev_value)),
    'ho': tf.Variable(tf.random_normal([n_hidden, vocab_size], stddev=stddev_value))
}
biases = {
    'we': tf.Variable(tf.random_normal([embedding_size], stddev=stddev_value)),
    'ho': tf.Variable(tf.random_normal([vocab_size], stddev=stddev_value))
}

# Calculate the network outputs
with tf.variable_scope('LSTM1'):
    pred = RNN(init_tokens)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=init_tokens))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# ===============================  Graph  ====================================

# Initialization
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step < training_iters:
        batch_tokens = next_batch(all_data, vocab_size)
        sess.run(optimizer, feed_dict={init_tokens: batch_tokens})

        if step % 1000 == 0:
            # loss = sess.run(cost, feed_dict={init_tokens: batch_tokens})
            _, loss, prediction, groundtruth = sess.run([optimizer, cost, pred, init_tokens],
                                                        feed_dict={init_tokens: batch_tokens})
            print("Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss))
            print(prediction[0:20, :].argmax(1))
            print(groundtruth[0:20, :].argmax(1))
        step += 1
    print("Optimization Completed!")
