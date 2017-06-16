
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


# ## Parameters

# ## Helper functions

# In[23]:

# data from: http://ai.stanford.edu/~amaas/data/sentiment/
TRAIN_INPUT = 'data/train.csv'
TEST_INPUT = 'data/test.csv'

# data manually generated
MY_TEST_INPUT = 'data/mytest.csv'

# Parameters for training
STEPS = 200
BATCH_SIZE = 64

# Parameters for data processing
SEQUENCE_LENGTH_KEY = 'sequence_length'
REVIEW_KEY = 'review'
CLASSIFICATION_KEY = 'is_positive'

# Vocabulary size
VOCAB_FILE = 'data/vocab.txt'
VOCAB = [line[:len(line)-1] for line in open(VOCAB_FILE)]
VOCAB_SIZE = len(VOCAB) - 1
print('there are %s words in the train and test files' % VOCAB_SIZE)

# In[24]:

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
                           dense_shape=sp.dense_shape,
                           )

def array_to_onehot(array, num_dim=2):
    array = np.asarray(array, dtype=np.int32)
    onehot = np.zeros([array.shape[0], num_dim])
    for i in range(array.shape[0]):
        onehot[i][array[i]] = 1
    return onehot

# Returns the column values from a CSV file as a list
def _get_csv_column(csv_file, column_name):
    with open(csv_file, 'r') as f:
        df = pd.read_csv(f)
        return df[column_name].tolist()


# ## Input functions

# In[25]:

# Input function used for training and testing                                                
def get_input_fn(csv_file, batch_size, epochs=1):
    with open(csv_file, 'r') as f:
        df = pd.read_csv(f)
        
        def input_fn():
            # Using queue with multiple threads to make it scalable
            pandas_queue = ff._enqueue_data(df,
                                            capacity=1024,
                                            shuffle=True,
                                            min_after_dequeue=256,
                                            num_threads=4,
                                            enqueue_size=16,
                                            num_epochs=epochs)

            _, review, classification, seq_len = pandas_queue.dequeue_up_to(batch_size)
            
            # Split sentences into words
            split_review = tf.string_split(review, delimiter=' ')
            # Creating a tf constant to hold the word -> index
            # this is need to create the sparse tensor and after the one hot encode
            mapping = tf.constant(VOCAB, name="mapping")
            # Words represented in a sparse tensor
            integerized_review = _sparse_string_to_index(split_review, mapping)

            # Converting numbers to int 32
            classification = tf.cast(classification, tf.int32)
            seq_len = tf.cast(seq_len, tf.int32)
            
            # Generates batcheds
            batched = tf.train.shuffle_batch({REVIEW_KEY: integerized_review,
                                              SEQUENCE_LENGTH_KEY: seq_len,
                                              CLASSIFICATION_KEY: classification},
                                             batch_size,
                                             min_after_dequeue=100,
                                             num_threads=4,
                                             capacity=1000,
                                             enqueue_many=True,
                                             allow_smaller_final_batch=True)
            
            label = batched.pop(CLASSIFICATION_KEY)
            label_onehot = tf.one_hot(label, 2)
            return batched, label_onehot
    return input_fn

# Creating my own input function for a custom CSV file
# it's simpler than the input_fn above but just used for small tests
def get_my_input_fn():
    def _input_fn():
        with open(MY_TEST_INPUT, 'r') as f:
            df = pd.read_csv(f)

            review = df.review.tolist()

            # Split sentences into words
            split_review = tf.string_split(review, delimiter=' ')
            # Creating a tf constant to hold the word -> index
            # this is need to create the sparse tensor and after the one hot encode
            mapping = tf.constant(VOCAB, name="mapping")
            # Words represented in a sparse tensor
            integerized_review = _sparse_string_to_index(split_review, mapping)
            
            x = {REVIEW_KEY: integerized_review, SEQUENCE_LENGTH_KEY: df.sequence_length.tolist()}

            y = df.is_positive.tolist()

            return x, y
    return _input_fn


# In[26]:

train_input_fn = get_input_fn(TRAIN_INPUT, BATCH_SIZE, None)
test_input_fn = get_input_fn(TEST_INPUT, BATCH_SIZE)
my_test_input_fn = get_my_input_fn()


# In[27]:

'''
# Testing the input function
with tf.Graph().as_default():
    train_input = train_input_fn()
    with tf.train.MonitoredSession() as sess:
        print (train_input)
        print (sess.run(train_input))
'''


# ## Creating the Estimator model

# In[33]:

def get_model_fn(rnn_cell_sizes,
                 label_dimension,
                 dnn_layer_sizes=[],
                 optimizer='SGD',
                 learning_rate=0.01,
                 embed_dim=128):
    
    def model_fn(features, labels, mode):
        
        review = features[REVIEW_KEY]
        sequence_length = tf.cast(features[SEQUENCE_LENGTH_KEY], tf.int32)
        labels_onehot = labels
        
        # Creating dense representation for the sentences
        # and then converting it to embeding representation
        dense_review = tf.sparse_tensor_to_dense(review, default_value=VOCAB_SIZE)
        embed_review = tf.contrib.layers.embed_sequence(dense_review,
                                                        vocab_size=(VOCAB_SIZE + 1),
                                                        embed_dim=embed_dim)
        
        
        # Each RNN layer will consist of a LSTM cell
        rnn_layers = [tf.contrib.rnn.LSTMCell(size) for size in rnn_cell_sizes]
        
        # Construct the layers
        multi_rnn_cell = tf.contrib.rnn.MultiRNNCell(rnn_layers)
        
        # Runs the RNN model dynamically
        # more about it at: 
        # https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn
        outputs, final_state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                                 inputs=embed_review,
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
        predictions_softmax = tf.nn.softmax(predictions)
        
        
        
        loss = None
        train_op = None
        
        preds_op = {
            'prediction': predictions_softmax,
            'label': labels_onehot
        }
        
        eval_op = {
            "accuracy": tf.metrics.accuracy(
                     tf.argmax(input=predictions_softmax, axis=1),
                     tf.argmax(input=labels_onehot, axis=1))
        }
        
        if mode != tf.contrib.learn.ModeKeys.INFER:    
            loss = tf.losses.softmax_cross_entropy(labels_onehot, predictions)
    
        if mode == tf.contrib.learn.ModeKeys.TRAIN:    
            train_op = tf.contrib.layers.optimize_loss(
              loss,
              tf.contrib.framework.get_global_step(),
              optimizer=optimizer,
              learning_rate=learning_rate)
        
        return tf.contrib.learn.ModelFnOps(mode,
                                           predictions=predictions_softmax,
                                           loss=loss,
                                           train_op=train_op,
                                           eval_metric_ops=eval_op)
    return model_fn


# In[34]:

model_fn = get_model_fn(rnn_cell_sizes=[256], # size of the hidden layers
                        label_dimension=2, # since are just 2 classes
                        dnn_layer_sizes=[128], # size of units in the dense layers on top of the RNN
                        optimizer='Adam',
                        learning_rate=0.001)
estimator = tf.contrib.learn.Estimator(model_fn=model_fn, model_dir='tensorboard/')


# ## Create and Run Experiment

# In[35]:

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


# In[36]:

# run experiment 
learn_runner.run(generate_experiment_fn(), '/tmp/outputdir')


# ## Making Predictions

# In[22]:

preds = estimator.predict(input_fn=my_test_input_fn, as_iterable=True)

sentences = _get_csv_column(MY_TEST_INPUT, 'review')

print()
for p, s in zip(preds, sentences):
    print('sentence:', s)
    print('bad review:', p[0], 'good review:', p[1])
    print('-' * 10)

