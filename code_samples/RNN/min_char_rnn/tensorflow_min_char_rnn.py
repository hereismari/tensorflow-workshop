"""
Vanilla Char-RNN using TensorFlow by Vinh Khuc (@knvinh).
Adapted from Karpathy's min-char-rnn.py
https://gist.github.com/karpathy/d4dee566867f8291f086
Requires tensorflow>=1.0
BSD License

Original code from: Vinh Khuc (@knvinh)
Modified by: Marianne Linhares (@mari-linhares)
"""

import numpy as np
import tensorflow as tf


# helper functions
def one_hot(v):
  return np.eye(VOCAB_SIZE)[v]

# data I/O

# should be simple plain text file
data = open('data/about_tensorflow.txt', 'r').read()
chars = list(set(data))

DATA_SIZE, VOCAB_SIZE = len(data), len(chars)
print('data has %d characters, %d unique.' % (DATA_SIZE, VOCAB_SIZE))

char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# hyperparameters
HIDDEN_SIZE = 100  # hidden layer's size
SEQ_LENGTH = 25   # number of steps to unroll
LEARNING_RATE = 1e-1  # size of step for Gradient Descent

# TensorFlow graph definition

# Placeholders
inputs = tf.placeholder(shape=[None, VOCAB_SIZE], dtype=tf.float32,
                        name='inputs')
targets = tf.placeholder(shape=[None, VOCAB_SIZE], dtype=tf.float32,
                         name='targets')
init_state = tf.placeholder(shape=[1, HIDDEN_SIZE], dtype=tf.float32,
                            name='state')

# Random initializer will be used by all variables
initializer = tf.random_normal_initializer(stddev=0.01)


# Initialize the RNN
def get_RNN_variables(reuse):
  with tf.variable_scope('RNN', reuse=reuse):
    # hidden layer
    Wxh = tf.get_variable('Wxh', [VOCAB_SIZE, HIDDEN_SIZE],
                          initializer=initializer)
    Whh = tf.get_variable('Whh', [HIDDEN_SIZE, HIDDEN_SIZE],
                          initializer=initializer)
    bh = tf.get_variable('bh', [HIDDEN_SIZE],
                         initializer=initializer)

    # output layer
    Why = tf.get_variable('Why', [HIDDEN_SIZE, VOCAB_SIZE],
                          initializer=initializer)
    by = tf.get_variable('by', [VOCAB_SIZE],
                         initializer=initializer)

  return Wxh, Whh, bh, Why, by

# Train the rnn

# defining the initial state of the RNN
hs_t = init_state

# initialize RNN variables
get_RNN_variables(reuse=False)

# getting RNN variables
Wxh, Whh, bh, Why, by = get_RNN_variables(reuse=True)

# Keep track of all outputs
ys = []
for t, xs_t in enumerate(tf.split(inputs, SEQ_LENGTH, axis=0)):
  # hidden layer output
  hs_t = tf.tanh(tf.matmul(xs_t, Wxh) + tf.matmul(hs_t, Whh) + bh)

  # final output
  ys_t = tf.matmul(hs_t, Why) + by
  ys.append(ys_t)

# Saves last hidden state
hprev = hs_t
# Concat outputs
outputs = tf.concat(ys, axis=0)
# Loss
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=targets,
                                                             logits=outputs))

# Minimizer
minimizer = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE,
                                      use_locking=True)
# Here's where the magic happens!
grads_and_vars = minimizer.compute_gradients(loss)

# Gradient clipping
grad_clipping = tf.constant(5.0, name='grad_clipping')
clipped_grads_and_vars = []
for grad, var in grads_and_vars:
  clipped_grad = tf.clip_by_value(grad, -grad_clipping, grad_clipping)
  clipped_grads_and_vars.append((clipped_grad, var))

# Gradient updates
# more magic!!!
updates = minimizer.apply_gradients(clipped_grads_and_vars)

# Sampling
Wxh, Whh, bh, Why, by = get_RNN_variables(reuse=True)
h_sample = tf.tanh(tf.matmul(inputs, Wxh) + tf.matmul(init_state, Whh) + bh)
y_sample = tf.matmul(h_sample, Why) + by
pred_sample = tf.nn.softmax(y_sample)

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
    hprev_val = np.zeros([1, HIDDEN_SIZE])  # reset RNN memory
    p = 0  # go from start of data

  input_vals = [char_to_ix[ch] for ch in data[p:p + SEQ_LENGTH]]
  target_vals = [char_to_ix[ch] for ch in data[p + 1:p + SEQ_LENGTH + 1]]

  input_vals_onehot = one_hot(input_vals)
  target_vals_onehot = one_hot(target_vals)

  # sampling
  if n % 1000 == 0:
    sample_length = 200
    hprev_sample = np.copy(hprev_val)

    # start from the first letter from the input
    x = np.zeros((VOCAB_SIZE, 1))
    x[input_vals[0]] = 1

    # stores predictions
    sample_ix = []
    for t in range(sample_length):
      # reshaping so it has the same shape as TensorFlow input
      x = np.reshape(x, [-1, VOCAB_SIZE])

      pred, hprev_sample = sess.run([pred_sample, h_sample],
                                    feed_dict={inputs: x,
                                               init_state: hprev_sample})
      # generates next letter
      ix = np.random.choice(range(VOCAB_SIZE), p=pred.ravel())
      # update next char with the prediction
      x = np.zeros((VOCAB_SIZE, 1))
      x[ix] = 1
      sample_ix.append(ix)

    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('----\n%s\n----' % (txt))

  # training
  hprev_val, loss_val, _ = sess.run([hprev, loss, updates],
                                    feed_dict={inputs: input_vals_onehot,
                                               targets: target_vals_onehot,
                                               init_state: hprev_val})

  # print progress
  smooth_loss = smooth_loss * 0.999 + loss_val * 0.001
  if n % 100 == 0:
    print('iter %d, loss: %f' % (n, smooth_loss))

  p += SEQ_LENGTH
  n += 1
