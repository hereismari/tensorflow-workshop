
# coding: utf-8

# ## Dependencies

# In[1]:

# Tensorflow
import tensorflow as tf

# Feeding function for enqueue data
# 
from tensorflow.python.estimator.inputs.queues import feeding_functions as ff

# Rnn common functions
from tensorflow.contrib.learn.python.learn.estimators import rnn_common

# Run an experiment
from tensorflow.contrib.learn.python.learn import learn_runner

# Input function
from tensorflow.python.estimator.inputs import numpy_io

# Helpers for data processing
import pandas as pd
import numpy as np

# Plot images with pyplot
from matplotlib import pyplot as plt

# ## Parameters

# In[2]:

# Data files
TRAIN_INPUT = 'data/train.csv'
TEST_INPUT = 'data/test.csv'
MY_TEST_INPUT = 'data/mytest.csv'

# Parameters for training
STEPS = 10000
BATCH_SIZE = 64

# Parameters for data processing
CHARACTERS = [chr(i) for i in range(256)]

SEQUENCE_LENGTH_KEY = 'sequence_length'
COLOR_NAME_KEY = 'color_name'
RGB_KEY = 'rgb'


# ## Helper functions

# In[3]:

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
    return tf.SparseTensor(indices=sp.indices,
                           values=tf.contrib.lookup.string_to_index(sp.values,
                                                                    mapping),
                           dense_shape=sp.dense_shape)

# Returns the column values from a CSV file as a list
def _get_csv_column(csv_file, column_name):
    with open(csv_file, 'r') as f:
        df = pd.read_csv(f)
        return df[column_name].tolist()

# Plot a color image
def _plot_rgb(rgb):
    data = [[rgb]]
    plt.figure(figsize=(2,2))
    plt.imshow(data, interpolation='nearest')
    plt.show()


# ## Input functions

# In[4]:

# Input function used for training and testing                                                
def get_input_fn(csv_file, batch_size, epochs=1):
    with open(csv_file, 'r') as f:
        df = pd.read_csv(f)
        
        # Sequence length is used by the Dynamic RNN
        # to dynamically unroll the graph :D!
        df['sequence_length'] = df.name.str.len().astype(np.int32)

        def input_fn():
            # Using queue with multiple threads to make it scalable
            pandas_queue = ff._enqueue_data(df,
                                            capacity=1024,
                                            shuffle=True,
                                            min_after_dequeue=256,
                                            num_threads=4,
                                            enqueue_size=16,
                                            num_epochs=epochs)

            _, color_name, r, g, b, seq_len = pandas_queue.dequeue_up_to(batch_size)

            # Split strings into chars
            split_color_name = tf.string_split(color_name, delimiter='')
            # Creating a tf constant to hold the map char -> index
            # this is need to create the sparse tensor and after the one hot encode
            mapping = tf.constant(CHARACTERS, name="mapping")
            # Names represented in a sparse tensor
            integerized_color_name = _sparse_string_to_index(split_color_name, mapping)

            # Tensor of normalized RGB values
            rgb = tf.to_float(tf.stack([r, g, b], axis=1)) / 255.0

            # Generates batcheds
            batched = tf.train.shuffle_batch({COLOR_NAME_KEY: integerized_color_name,
                                              SEQUENCE_LENGTH_KEY: seq_len,
                                              RGB_KEY: rgb},
                                             batch_size,
                                             min_after_dequeue=100,
                                             num_threads=4,
                                             capacity=1000,
                                             enqueue_many=True,
                                             allow_smaller_final_batch=True)
            label = batched.pop(RGB_KEY)
            return batched, label
    return input_fn

# Creating my own input function for a custom CSV file
# it's simpler than the input_fn above but just used for small tests
def get_my_input_fn():
    def _input_fn():
        with open(MY_TEST_INPUT, 'r') as f:
            df = pd.read_csv(f)
            df['sequence_length'] = df.name.str.len().astype(np.int32)

            color_name = df.name.tolist()
    
            split_color_name = tf.string_split(color_name, delimiter='')
            mapping = tf.constant(CHARACTERS, name="mapping")
            integerized_color_name = _sparse_string_to_index(split_color_name, mapping)

            x = {COLOR_NAME_KEY: integerized_color_name, SEQUENCE_LENGTH_KEY: df.sequence_length.tolist()}

            y = np.asarray([[0, 0, 0]], dtype=np.float32)

            return x, y
    return _input_fn


# In[5]:

train_input_fn = get_input_fn(TRAIN_INPUT, BATCH_SIZE, None)
test_input_fn = get_input_fn(TEST_INPUT, BATCH_SIZE)
my_test_input_fn = get_my_input_fn()


# In[6]:

'''
# Testing the input function
with tf.Graph().as_default():
    train_input = train_input_fn()
    with tf.train.MonitoredSession() as sess:
        print (train_input)
        print (sess.run(train_input))
'''


# ## Creating the Estimator model

# In[7]:

def get_model_fn(rnn_cell_sizes,
                 label_dimension,
                 dnn_layer_sizes=[],
                 optimizer='SGD',
                 learning_rate=0.01):
    
    def model_fn(features, labels, mode):
        
        color_name = features[COLOR_NAME_KEY]
        sequence_length = features[SEQUENCE_LENGTH_KEY]

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
        
        return tf.contrib.learn.ModelFnOps(mode,
                                           predictions=predictions,
                                           loss=loss,
                                           train_op=train_op)
    return model_fn


# In[10]:

model_fn = get_model_fn(rnn_cell_sizes=[256, 128], # size of the hidden layers
                        label_dimension=3, # since is RGB
                        dnn_layer_sizes=[64], # size of units in the dense layers on top of the RNN
                        optimizer='Adam',
                        learning_rate=0.01)
estimator = tf.contrib.learn.Estimator(model_fn=model_fn)


# ## Create and Run Experiment

# In[11]:

# create experiment
def generate_experiment_fn():
  
  """
  Create an experiment function given hyperparameters.
  Returns:
    A function (output_dir) -> Experiment where output_dir is a string
    representing the location of summaries, checkpoints, and exports.
    this function is used by learn_runner to create an Experiment which
    executes model code provided in the form of an Estimator and
    input functions.
    All listed arguments in the outer function are used to create an
    Estimator, and input functions (training, evaluation, serving).
    Unlisted args are passed through to Experiment.
  """

  def _experiment_fn(output_dir):
    return tf.contrib.learn.Experiment(
        estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=test_input_fn,
        train_steps=STEPS
    )
  return _experiment_fn


# In[12]:

# run experiment 
learn_runner.run(generate_experiment_fn(), '/tmp/outputdir')


# ## Making Predictions

# In[13]:

p2 = estimator.predict(input_fn=test_input_fn, as_iterable=True)

for i in range(5):
    print(next(p2) * 255)


# ## Visualizing the result

# In[46]:

preds = estimator.predict(input_fn=my_test_input_fn, as_iterable=True)

color_names = _get_csv_column(MY_TEST_INPUT, 'name')

print()
for p, name in zip(preds, color_names):
    color = tuple(map(int, p * 255))
    print(name, 'rgb', color)
    _plot_rgb(p)

