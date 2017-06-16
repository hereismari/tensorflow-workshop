import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os.path

data = open('data/about_tensorflow.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 2e-1
batch_size = 10
num_epochs = 500

# Convert an array of chars to array of vocab indices
def c2i(inp):
    return map(lambda c:char_to_ix[c], inp)

def i2c(inp):
    return map(lambda c:ix_to_char[c], inp)

# Generate data for an epoch, with batches of size batch_size.
def gen_epoch_data(raw_data, batch_size):
    data_len = len(raw_data)
    num_examples = (data_len - 1) // seq_length
    num_batches = num_examples // batch_size

    epoch_data = []
    for i in range(num_batches):
        batch = []
        idx = i * batch_size * seq_length
        for j in range(batch_size):
            inp = raw_data[idx + j*seq_length:idx + (j+1)*seq_length]
            target = raw_data[idx + 1+(j*seq_length):idx + 1+((j+1)*seq_length)]

            batch.append([c2i(inp), c2i(target)])
        epoch_data.append(batch)
    return epoch_data


epoch_data = gen_epoch_data(data, batch_size)
init_state = tf.zeros([hidden_size, 1])

# Input
x = tf.placeholder(tf.int32, shape=(seq_length), name="x")
y = tf.placeholder(tf.int32, shape=(seq_length), name="y")
state = tf.zeros([hidden_size, 1])

# One Hot representation of the input
x_oh = tf.one_hot(indices=x, depth=vocab_size)
y_oh = tf.one_hot(indices=y, depth=vocab_size)

rnn_inputs = tf.unstack(x_oh)
rnn_targets = tf.unstack(y_oh)

# Setup the weights and biases.
with tf.variable_scope('rnn_cell'):
    Wxh = tf.get_variable('Wxh', [hidden_size, vocab_size])
    Whh = tf.get_variable('Whh', [hidden_size, hidden_size])
    Why = tf.get_variable('Why', [vocab_size, hidden_size])
    bh = tf.get_variable('bh', [hidden_size, 1])
    by = tf.get_variable('by', [vocab_size, 1])

# Actual math behind computing the output and the next state of the RNN.
def rnn_cell(rnn_input, cur_state):
    with tf.variable_scope('rnn_cell', reuse=True):
        Wxh = tf.get_variable('Wxh', [hidden_size, vocab_size])
        Whh = tf.get_variable('Whh', [hidden_size, hidden_size])
        Why = tf.get_variable('Why', [vocab_size, hidden_size])
        bh = tf.get_variable('bh', [hidden_size, 1])
        by = tf.get_variable('by', [vocab_size, 1])
    inp = tf.expand_dims(rnn_input, 1)

    next_state = tf.tanh(tf.matmul(Wxh, inp) + tf.matmul(Whh, cur_state) + bh)
    y_hat = tf.matmul(Why, next_state) + by
    return y_hat, next_state

logits = []
for rnn_input in rnn_inputs:
    y_hat, state = rnn_cell(rnn_input, state)
    y_hat = tf.squeeze(y_hat)
    logits.append(y_hat)

losses = [tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=target) for logit, target in zip(logits, rnn_targets)]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(total_loss)

num_samples = 25
sample_state = init_state
seed = tf.placeholder(tf.int32, [1], name='seed')
rnn_input = tf.one_hot(seed, vocab_size)
ixes = []

rnn_input = tf.squeeze(rnn_input)
y_hat, sample_state = rnn_cell(rnn_input, sample_state)
prob = tf.nn.softmax(tf.squeeze(y_hat))


def train():
    tlosses = []
    saver = tf.train.Saver()
    with tf.Session() as sess:
        if os.path.isfile("model.ckpt"):
            saver.restore(sess, "model.ckpt")
        else:
            sess.run(tf.initialize_all_variables())

        for epoch_idx in range(num_epochs):
            print '--- Starting Epoch:', epoch_idx, '---'
            epoch_loss = 0
            epoch_state = np.zeros([hidden_size, 1])
            equals = 0.0
            for idx, batch in enumerate(epoch_data):
                training_loss = 0
                for example_idx, example in enumerate(batch):
                    x_i = example[0]
                    y_i = example[1]

                    loss, tloss, _, logits_, rnn_targets_, epoch_state = \
                        sess.run([losses, total_loss, train_step, logits, \
                            rnn_targets, state], \
                                feed_dict={x:x_i, y:y_i, init_state:epoch_state}
                        )

                    logits_argmax = np.argmax(logits_, axis=1)
                    rnn_targets_argmax = np.argmax(rnn_targets_, axis=1)
                    equals += np.sum(logits_argmax == rnn_targets_argmax)

                    training_loss += tloss

                    if (example_idx % 10 == 0):
                        inp_seed = np.array([example[0][0]])

                        print '\n'
                        print '--- SAMPLE BEGIN ---'
                        num_chars = 100
                        ixes = []
                        sstate = np.zeros([hidden_size, 1])
                        for j in range(num_chars):
                            prob_r, sstate = sess.run([prob, sample_state], feed_dict={seed:inp_seed, init_state:sstate, x:x_i})
                            ix = np.random.choice(vocab_size, p=prob_r.ravel())
                            ixes.append(ix)
                            inp_seed = np.array([ix])

                        print ''.join(i2c(ixes))
                        print '--- SAMPLE END ---'

                training_loss /= len(batch)
                equals /= len(batch)
                print 'Epoch:', epoch_idx, 'Batch:', idx
                print 'Average training loss in batch:', training_loss
                print 'Average matching chars per batch:', equals
                tlosses.append(training_loss)
            save_path = saver.save(sess, "model.ckpt")
            print("Model saved in file: %s" % save_path)
    return tlosses


tlosses = train()
plt.plot(tlosses)
plt.show()
