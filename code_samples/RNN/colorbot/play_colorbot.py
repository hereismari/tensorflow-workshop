# coding: utf-8

# Tensorflow
import tensorflow as tf
# Rnn common functions
from tensorflow.contrib.learn.python.learn.estimators import rnn_common
# Model builder
from tensorflow.python.estimator import model_fn as model_fn_lib
print('Tested with TensorFLow 1.2.0')
print('Your TensorFlow version:', tf.__version__) 
from tensorflow.python.ops import math_ops

# Plot images with pyplot
from matplotlib import pyplot as plt

# Helpers for data processing
import pandas as pd
import numpy as np
import argparse

# parser definition
import argparse

parser = argparse.ArgumentParser(prog='Play with Colorbot!')

parser.add_argument('--model_path', type=str, default='pretrained',
		    help='Local path to the folder where the colorbot'
				 'model is.')

# ## Helper functions

# In[170]:

# This function creates a sparse tensor in the following way, given:
# indices = [[0, 0], [1, 1], [2, 2]]
# values = [1, 2, 3]
# dense_shape = [3, 4]
#
# The output will be a sparse tensor that represents this dense tensor:
# [ 
#   [1, 0, 0, 0]
#   [0, 2, 0, 0]
#   [0, 0, 3, 0]
# ]
#
# We're using this to generate a Sparse tensor that can be easily
# formated in a one hot representation.
# More at: https://www.tensorflow.org/api_docs/python/tf/SparseTensor
def _sparse_string_to_index(sp, mapping):
    # This operation constructs a lookup table to convert tensor of strings
    # into int64 IDs
    table = tf.contrib.lookup.index_table_from_tensor(mapping, dtype=tf.string)
    
    return tf.SparseTensor(indices=sp.indices,
                           values=table.lookup(sp.values),
                           dense_shape=sp.dense_shape)

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
RGB_KEY = 'rgb'

# Creating my own input function for a given color
def get_input_fn(color):
	def input_fn():
		seq_len = list([len(color)])
		#x = math_ops.to_int32(seq_len)
		#print(x.get_shape().ndims)
		color_name = [color] # the input for string_split needs to be
						     # a tensor

		split_color_name = tf.string_split(color_name, delimiter='')
		mapping = tf.constant(CHARACTERS, name="mapping")
		integerized_color_name = _sparse_string_to_index(split_color_name, mapping)

		# generating anything (0, 0, 0) for the y
		# since for most cases there's no right answer 
		y = np.asarray([[0, 0, 0]], dtype=np.float32)

		# creates inputs
		x = {COLOR_NAME_KEY: integerized_color_name,
             SEQUENCE_LENGTH_KEY: seq_len}
                                          
		return x, y
	return input_fn

# Loading the Estimator model
def get_model_fn(rnn_cell_sizes,
                 label_dimension,
                 dnn_layer_sizes=[],
                 optimizer='SGD',
                 learning_rate=0.01):
    
    def model_fn(features, labels, mode):
        
        color_name = features[COLOR_NAME_KEY]
        sequence_length = features[SEQUENCE_LENGTH_KEY]
        print(sequence_length)
        x = math_ops.to_int32(sequence_length)
        print(x.get_shape().ndims)

        # Creating dense representation for the names
        # and then converting it to one hot representation
        dense_color_name = tf.sparse_tensor_to_dense(color_name, default_value=len(CHARACTERS))
        color_name_onehot = tf.one_hot(dense_color_name, depth=len(CHARACTERS) + 1)
        
        # Each RNN layer will consist of a LSTM cell
        rnn_layers = [tf.contrib.rnn.LSTMCell(size) for size in rnn_cell_sizes]
        
        # Construct the layers
        multi_rnn_cell = tf.contrib.rnn.MultiRNNCell(rnn_layers)
        
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

        if mode != tf.contrib.learn.ModeKeys.INFER:    
            loss = tf.losses.mean_squared_error(labels, predictions)
    
        if mode == tf.contrib.learn.ModeKeys.TRAIN:    
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

model_fn = get_model_fn(rnn_cell_sizes=[256, 128], # size of the hidden layers
                        label_dimension=3, # since is RGB
                        dnn_layer_sizes=[128], # size of units in the dense layers on top of the RNN
                        optimizer='Adam', #changing optimizer to Adam
                        learning_rate=0.01)

args = parser.parse_args()
estimator = tf.estimator.Estimator(model_fn=model_fn,
								   model_dir=args.model_path)


# Making Predictions
print('Colorbot is ready to generate colors!')

EXIT_COMMAND = '<exit>'
while True:
	color_name = input('give me a color name (or %s): ' % (EXIT_COMMAND))
	if color_name == EXIT_COMMAND:
		break
	
	print('Generating color...')
	preds = estimator.predict(input_fn=get_input_fn(color_name))
	for p, name in zip(preds, [color_name]):
		color = tuple(map(int, p * 255))
		_plot_rgb(p, name)
