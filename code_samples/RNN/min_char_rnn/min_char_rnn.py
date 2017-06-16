"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License

Original code from: Andrej Karpathy (@karpathy)
Modified by: Marianne Linhares (@mari-linhares)
"""

import numpy as np

# helper functions
def softmax(x):
  return np.exp(x) / np.sum(np.exp(x))

# data I/O
data = open('data/about_tensorflow.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
DATA_SIZE, VOCAB_SIZE = len(data), len(chars)
print('data has %d characters, %d unique.' % (DATA_SIZE, VOCAB_SIZE))
char_to_ix = { ch : i for i, ch in enumerate(chars) }
ix_to_char = { i: ch for i, ch in enumerate(chars) }

# hyperparameters
HIDDEN_SIZE = 100 # size of hidden layer of neurons
SEQ_LENGTH = 25 # number of steps to unroll the RNN for (static RNN)
LEARNING_RATE = 1e-1 # size of step for Gradient Descent

# model parameters
# x: input, h: hidden, y: output
# W: weight, b: bias
Wxh = np.random.randn(HIDDEN_SIZE, VOCAB_SIZE)* 0.01 # weight: input to hidden
Whh = np.random.randn(HIDDEN_SIZE, HIDDEN_SIZE)*0.01 # weight: hidden to hidden
Why = np.random.randn(VOCAB_SIZE, HIDDEN_SIZE)*0.01 # weight: hidden to output

bh = np.zeros((HIDDEN_SIZE, 1)) # hidden bias
by = np.zeros((VOCAB_SIZE, 1)) # output bias

# loss
def lossFun(inputs, targets, hprev):
  """
  inputs, targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0

  # forward pass
  for t in range(len(inputs)):
    # encode in 1-of-k representation
    xs[t] = np.zeros((VOCAB_SIZE, 1))
    xs[t][inputs[t]] = 1

    # hidden state
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)

    # unnormalized log probabilities for next chars
    ys[t] = np.dot(Why, hs[t]) + by

    # probabilities for next chars
    ps[t] = softmax(ys[t])

    # softmax (cross-entropy loss)
    loss += -np.log(ps[t][targets[t],0])

  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)

  # this will keep track of the hidden state derivative
  dhnext = np.zeros_like(hs[0])

  # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
  # for the comments bellow consider:
  # ps[t] = softmax(A[t])
  # A[t] = Why * hs[t] + by
  # hs[t] = tanh(B)
  # B = Whh * hs[t-1] + Wxh * x[t] + bh
  for t in reversed(range(len(inputs))):
    # backprop into the output
    # derivative of the softmax function = probs - 1
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1
    
    # calculating derivative for weight: hidden to output
    # dWhy[t] = (dSoftmax / dA) * (dA / dWhy) = dy * hs[t].T
    dWhy += np.dot(dy, hs[t].T) 
    # dWhy[t] = (dSoftmax / dA) * (dA / dby) = dy * 1
    dby += dy # calculating derivative for the output bias

    # backprop into hidden state
    
    # calculating derivative for hidden state
    # IMPORTANT: why we're adding two values?
    # the gradients for the hidden state are composed of two parts
    # a vertical output and a horizontal output, think about it...
    # dvert:
    #     keeps track of the vertical gradient
    # dhor:
    #     keeps track of the horizontal gradient
    #     which is calculated multiplying the dhraw by the Whh
    #     think about it...
    
    # dvert[t] = (dSoftmax / dA) * (dA / dh[t]) = dy * Why
    dvert = np.dot(Why.T, dy)
    dhor = dhnext
	
	# dh[t] = dvert[t] + dhor[t]
    dh = dvert + dhor
    
    # derivative of tanh, f'(z) = 1 - f(z) ** 2
    # dhraw = dB[t] = (dvert + dhor) * (dh[t] / dB) 
    # dhraw = (dvert + dhor) * derivative(tanh(B))
    # dhraw = dh * (1 - hs[t] ** 2) 
    dhraw = (1 - hs[t] * hs[t]) * dh 
    
    # dbh[t] = dhraw * (dB/dbh) = dhraw * 1
    dbh += dhraw 
    # dWxh[t] = dhraw * (dB/dWxh) = dhraw * xs[t]
    dWxh += np.dot(dhraw, xs[t].T)
    # dbh[t] = dhraw * (dB/dWhh) = dhraw * hs[t-1]
    dWhh += np.dot(dhraw, hs[t-1].T)

	# this keeps track of the horizontal gradient
	# which is the derivative of the next state
	# dhnext[t] = dhraw * (dB/dhs[t]) = dhraw * Whh
    dhnext = np.dot(Whh.T, dhraw) 
    
  # clip values between [-5, 5] to mitigate exploding gradients
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam)

  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """
  sample a sequence of integers from the model
  h is memory state, seed_ix is seed letter for first time step
  """
  # stores the current letter in one-hot format
  x = np.zeros((VOCAB_SIZE, 1))
  x[seed_ix] = 1
  # stores predictions
  ixes = []
  for t in range(n):
    # running RNN
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    # generating prob
    p = softmax(y)
    ix = np.random.choice(range(VOCAB_SIZE), p=p.ravel())
    # update next char with the prediction
    x = np.zeros((VOCAB_SIZE, 1))
    x[ix] = 1
    ixes.append(ix)
  # return predictions
  return ixes

# training

# n: counts the number of steps, p: pointer to position in sequence
n, p = 0, 0

# memory variables for Adagrad
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)

# loss at iteration 0
smooth_loss = -np.log(1.0 / VOCAB_SIZE) * SEQ_LENGTH

while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p + SEQ_LENGTH + 1 >= len(data) or n == 0:
    hprev = np.zeros((HIDDEN_SIZE, 1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p: p + SEQ_LENGTH]]
  targets = [char_to_ix[ch] for ch in data[p + 1: p + SEQ_LENGTH + 1]]

  # sample from the model now and then
  if n % 1000 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('----\n%s\n----' % (txt, ))

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  # print progress
  if n % 100 == 0: print('iter %d, loss: %f' % (n, smooth_loss))

  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                [dWxh, dWhh, dWhy, dbh, dby],
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -LEARNING_RATE * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += SEQ_LENGTH # move data pointer
  n += 1 # iteration counter
