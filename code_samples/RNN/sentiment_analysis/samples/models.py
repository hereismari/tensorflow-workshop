from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import rnn_common
import tensorflow.contrib.rnn as rnn


def get_model_fn(rnn_cell_sizes,
                 label_dimension,
                 num_words,
                 dnn_layer_sizes=[],
                 optimizer='SGD',
                 learning_rate=0.01,
                 embed_dim=30,
                 # if provided should have the same length as rnn_cell_sizes
                 # otherwise will not be used
                 dropout_keep_probabilities=[]):

  def model_fn(features, labels, mode):

    x = features['x']
    sequence_length = tf.cast(features[rnn_common.RNNKeys.SEQUENCE_LENGTH_KEY],
                              tf.int32)

    # creating embedding for the reviews
    embedding = tf.contrib.layers.embed_sequence(x,
                                                 vocab_size=num_words,
                                                 embed_dim=embed_dim)

    # Each RNN layer will consist of a LSTM cell
    if len(dropout_keep_probabilities) == len(rnn_cell_sizes):

      if mode != tf.estimator.ModeKeys.TRAIN:
        rnn_layers = [
            rnn.DropoutWrapper(rnn.LSTMCell(size),
                               input_keep_prob=1,
                               output_keep_prob=1,
                               state_keep_prob=1)
            for size, keep_prob in rnn_cell_sizes]
      else:
        rnn_layers = [
            rnn.DropoutWrapper(rnn.LSTMCell(size),
                               input_keep_prob=keep_prob,
                               output_keep_prob=keep_prob,
                               state_keep_prob=keep_prob)
            for size, keep_prob in zip(rnn_cell_sizes,
                                       dropout_keep_probabilities)]
    else:
      rnn_layers = [rnn.LSTMCell(size) for size in rnn_cell_sizes]

    # Construct the layers
    multi_rnn_cell = rnn.MultiRNNCell(rnn_layers)

    # Runs the RNN model dynamically
    # more about it at:
    # https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn
    outputs, final_state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                             inputs=embedding,
                                             sequence_length=sequence_length,
                                             dtype=tf.float32)

    # Slice to keep only the last cell of the RNN
    last_activations = rnn_common.select_last_activations(outputs,
                                                          sequence_length)

    # Construct dense layers on top of the last cell of the RNN
    for units in dnn_layer_sizes:
      last_activations = tf.layers.dense(last_activations,
                                         units,
                                         activation=tf.nn.relu)

    # Final dense layer for prediction
    predictions = tf.layers.dense(last_activations, label_dimension)
    predictions_softmax = tf.nn.softmax(predictions)

    loss = None
    train_op = None
    eval_op = None

    if mode != tf.estimator.ModeKeys.PREDICT:
      labels_onehot = tf.one_hot(labels, 2)

      eval_op = {
          'accuracy': tf.metrics.accuracy(
              tf.argmax(input=predictions_softmax, axis=1),
              tf.argmax(input=labels_onehot, axis=1))
      }

      loss = tf.losses.softmax_cross_entropy(labels_onehot, predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = tf.contrib.layers.optimize_loss(
          loss,
          tf.contrib.framework.get_global_step(),
          optimizer=optimizer,
          learning_rate=learning_rate)

    return tf.estimator.EstimatorSpec(mode,
                                      predictions=predictions_softmax,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_op)
  return model_fn

