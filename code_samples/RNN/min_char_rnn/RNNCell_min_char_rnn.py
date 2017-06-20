"""
Adapted from Karpathy's min-char-rnn.py
https://gist.github.com/karpathy/d4dee566867f8291f086
Requires tensorflow>=1.0
BSD License

Modified by: Marianne Linhares (@mari-linhares)
"""
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

print(tf.__version__)

# helper functions
def one_hot(v):
    return np.eye(VOCAB_SIZE)[v]

# data I/O
data = open('data/about_tensorflow.txt', 'r').read()
chars = list(set(data))
DATA_SIZE, VOCAB_SIZE = len(data), len(chars)
print('Data has %d characters, %d unique.' % (DATA_SIZE, VOCAB_SIZE))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# hyperparameters
HIDDEN_SIZE   = 100  # hidden layer's size
SEQ_LENGTH    = 25   # number of steps to unroll
LEARNING_RATE = 0.1 # size of step for Gradient Descent
BATCH_SIZE = 1 # number of sequences evaluated at each train step

# TensorFlow graph definition

# Placeholders
inputs     = tf.placeholder(shape=[None, None], dtype=tf.int32, name="inputs")
targets    = tf.placeholder(shape=[None, None], dtype=tf.int32, name="targets")
init_state = tf.placeholder(shape=[None, HIDDEN_SIZE], dtype=tf.float32, name="state")

# RNN

# [ BATCHSIZE, SEQLEN, ALPHASIZE ]
input_x = tf.one_hot(inputs, VOCAB_SIZE)
input_y = tf.one_hot(targets, VOCAB_SIZE)

# creating RNN cell
rnn_cell = tf.contrib.rnn.GRUCell(HIDDEN_SIZE)

# run RNN
rnn_outputs, final_state = tf.nn.dynamic_rnn(rnn_cell, input_x,
					     initial_state=init_state, dtype=tf.float32)
										  
rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, HIDDEN_SIZE]) # [ BATCHSIZE x SEQLEN, VOCAB_SIZE ]
dense_layer = layers.linear(rnn_outputs_flat, VOCAB_SIZE)
labels = tf.reshape(input_y, [-1, VOCAB_SIZE]) # [ BATCHSIZE x SEQLEN, VOCAB_SIZE ] 
output_softmax = tf.nn.softmax(dense_layer)   # [ BATCHSIZE x SEQLEN, ALPHASIZE ]

# Loss
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer, labels=labels))

# Minimizer
minimizer = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

# Gradient clipping
'''
# Here's where the magic happens!
grads_and_vars = minimizer.compute_gradients(loss)

grad_clipping = tf.constant(5.0, name="grad_clipping")
clipped_grads_and_vars = []
for grad, var in grads_and_vars:
    clipped_grad = tf.clip_by_value(grad, -grad_clipping, grad_clipping)
    clipped_grads_and_vars.append((clipped_grad, var))

# Gradient updates
# more magic!!!
updates = minimizer.apply_gradients(clipped_grads_and_vars)
'''
# Create session and initialize variables
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Training

# n: counts the number of steps, p: pointer to position in sequence
n, p = 0, 0

# this var will be used for sampling
hprev_val = np.zeros([1, HIDDEN_SIZE])

# loss at iteration 0
smooth_loss = -np.log(1.0 / VOCAB_SIZE) * SEQ_LENGTH

while True:
	
	# prepare inputs (we're sweeping from left to right in steps SEQ_LENGTH long)
	if p + SEQ_LENGTH + 1 >= len(data) or n == 0:
		hprev_val = np.zeros([BATCH_SIZE, HIDDEN_SIZE]) # reset RNN memory
		p = 0 # go from start of data

	input_vals  = np.reshape([char_to_ix[ch] for ch in data[p:p + SEQ_LENGTH]], (-1, SEQ_LENGTH))
	target_vals = np.reshape([char_to_ix[ch] for ch in data[p + 1:p + SEQ_LENGTH + 1]], (-1, SEQ_LENGTH))

	# sampling	
	if n % 1000 == 0:
		sample_length = 200
		hprev_sample = np.copy(hprev_val)
		
		# start from the first letter from the input
		first_char = char_to_ix[data[p]]
		x = np.asarray([[first_char]])
		# stores predictions
		sample_ix = []

		for t in range(sample_length):
			
			pred, hprev_sample = sess.run([output_softmax, final_state],
						      feed_dict={inputs: x, init_state: hprev_sample})
										 
			# generates next letter
			ix = np.random.choice(range(VOCAB_SIZE), p=pred.ravel())
			# update next char with the prediction
			x = np.asa rray([[ix]])
			sample_ix.append(ix)
			
		txt = ''.join(ix_to_char[ix] for ix in sample_ix)
		print('----\n%s\n----' % (txt, ))
	
	# running the graph
	hprev_val, loss_val, _ = sess.run([final_state, loss, minimizer],
					  feed_dict={inputs: input_vals,
						     targets: target_vals,
						     init_state: hprev_val})

	# print progress
	smooth_loss = smooth_loss * 0.999 + loss_val * 0.001
	if n % 100 == 0: print('iter %d, loss: %f' % (n, smooth_loss))

	p += SEQ_LENGTH
	n += 1
