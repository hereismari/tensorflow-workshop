# coding: utf-8

import argparse

from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import rnn_common
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.ops import math_ops


parser = argparse.ArgumentParser(prog='Play with Colorbot!')

parser.add_argument('--model_path', type=str, default='pretrained',
                    help='Local path to the folder where the colorbot'
                         'model is.')


# ## Helper functions
# Returns the column values from a CSV file as a list
def _get_csv_column(csv_file, column_name):
  with open(csv_file, 'r') as f:
    df = pd.read_csv(f)
    return df[column_name].tolist()


# Plot a color image
def _plot_rgb(rgb, color_name):
  data = [[rgb]]
  plt.imshow(data, interpolation='nearest')
  plt.title(color_name)
  plt.show()

# Helper variables for the input function
CHARACTERS = [chr(i) for i in range(256)]
SEQUENCE_LENGTH_KEY = 'sequence_length'
COLOR_NAME_KEY = 'color_name'


# Creating my own input function for a given color
def get_input_fn(color):
  def input_fn():
    seq_len = len(color)
    # color is now a sequence of chars
    color_split = tf.string_split([color], '').values

    # creating dataset
    dataset = tf.contrib.data.Dataset.from_tensors((color_split))
    # generating a batch, so it has the right rank
    dataset = dataset.batch(1)

    # creating iterator
    color_name = dataset.make_one_shot_iterator().get_next()

    features = {
        COLOR_NAME_KEY: color_name,
        SEQUENCE_LENGTH_KEY: [seq_len]
    }

    # we're just predicting, so the label can be None
    # if you're training make sure to return a label
    return features, None
  return input_fn


# Loading the Estimator model
def get_model_fn(rnn_cell_sizes,
                 label_dimension,
                 dnn_layer_sizes=[],
                 optimizer='SGD',
                 learning_rate=0.01):

  def model_fn(features, labels, mode):

    color_name = features[COLOR_NAME_KEY]
    # int64 -> int32
    sequence_length = tf.cast(features[SEQUENCE_LENGTH_KEY], dtype=tf.int32)

    # Creating a tf constant to hold the map char -> index
    # this is need to create the sparse tensor and after the one hot encode
    mapping = tf.constant(CHARACTERS, name='mapping')
    table = tf.contrib.lookup.index_table_from_tensor(mapping, dtype=tf.string)
    int_color_name = table.lookup(color_name)

    # representing colornames with one hot representation
    color_name_onehot = tf.one_hot(int_color_name, depth=len(CHARACTERS) + 1)

    # Each RNN layer will consist of a LSTM cell
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in rnn_cell_sizes]

    # Construct the layers
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    # Runs the RNN model dynamically
    # more about it at:
    # https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn
    outputs, final_state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                             inputs=color_name_onehot,
                                             sequence_length=sequence_length,
                                             dtype=tf.float32)

    # Slice to keep only the last cell of the RNN
    last_activations = rnn_common.select_last_activations(outputs,
                                                          sequence_length)

    # Construct dense layers on top of the last cell of the RNN
    for units in dnn_layer_sizes:
        last_activations = tf.layers.dense(
            last_activations, units, activation=tf.nn.relu)

    # Final dense layer for prediction
    predictions = tf.layers.dense(last_activations, label_dimension)

    loss = None
    train_op = None

    if mode != tf.estimator.ModeKeys.PREDICT:
      loss = tf.losses.mean_squared_error(labels, predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = tf.contrib.layers.optimize_loss(
          loss,
          tf.contrib.framework.get_global_step(),
          optimizer=optimizer,
          learning_rate=learning_rate)

    return model_fn_lib.EstimatorSpec(mode,
                                      predictions=predictions,
                                      loss=loss,
                                      train_op=train_op)
  return model_fn

model_fn = get_model_fn(rnn_cell_sizes=[256, 128],  # size of the hidden layers
                        label_dimension=3,  # since is RGB
                        dnn_layer_sizes=[128],
                        optimizer='Adam',  # changing optimizer to Adam
                        learning_rate=0.01)

args = parser.parse_args()
estimator = tf.estimator.Estimator(model_fn=model_fn,
                                   model_dir=args.model_path)


# Making Predictions
print('Colorbot is ready to generate colors!')

EXIT_COMMAND = '<exit>'
while True:
  color_name = raw_input('give me a color name (or %s): ' % (EXIT_COMMAND))
  if color_name == EXIT_COMMAND:
    break

  print('Generating color...')
  preds = estimator.predict(input_fn=get_input_fn(color_name))
  for p, name in zip(preds, [color_name]):
    color = tuple(map(int, p * 255))
    _plot_rgb(p, name)
