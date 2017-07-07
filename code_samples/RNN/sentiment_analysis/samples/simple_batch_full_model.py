from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# tensorflow
import tensorflow as tf  # tf v1.2 or higher
from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators.dynamic_rnn_estimator import PredictionType
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python.learn.estimators import rnn_common
from tensorflow.contrib.learn.python.learn import learn_runner  # run an experiment
print('TensorFlow version is', tf.__version__)

# data and data preprocessing
from tensorflow.contrib.keras.python.keras.datasets import imdb 
from tensorflow.contrib.keras.python.keras.preprocessing import sequence

import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir', type=str, default='sentiment_analysis_output',
    help='The directory where the model outputs should be stored')

FLAGS = parser.parse_args()


# Input function
def get_input_fn(x_in, y_in, batch_size,
                 shuffle=True, epochs=1, max_length=250):
        
  def input_fn():
    # calculates the length of the sequences, where
    # length = min(actual_length, MAX_LENGT
    x_len = np.minimum(np.array([len(seq) for seq in x_in]),
                       max_length).astype('int32')

    # x_post_pad = sequence.pad_sequences(x_in, maxlen=max_length, padding='post')
    x_post_pad = sequence.pad_sequences(x_in, maxlen=max_length, padding='post')   

    # creates the dataset from in memory data
    ds = tf.contrib.data.Dataset.from_tensor_slices((x_post_pad, x_len, y_in))
   
    # repeats the dataset `epochs` times.
    ds = ds.repeat(epochs)
    
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    ds = ds.batch(batch_size)
    
    # creates iterator
    x, x_len, y = ds.make_one_shot_iterator().get_next()

    dict_x = {'x': x, rnn_common.RNNKeys.SEQUENCE_LENGTH_KEY: x_len}
    return dict_x, y
        
  return input_fn


# create experiment
def generate_experiment_fn(x_train, y_train, x_test, y_test,
                           num_words,
                           max_length=250,
                           batch_size=16,
                           embedding_size=30,
                           learning_rate=0.001,
                           num_epochs=1,
                           cell_type='lstm',
                           optimizer='Adam',
                           num_units=[256, 128],
                           drouput_keep_probabilities=[0.6, 0.6, 0.6]):

  def _experiment_fn(run_config, hparams):

    # feature sequences
    xc = tf.contrib.layers.sparse_column_with_integerized_feature('x', num_words)
    xc = tf.contrib.layers.embedding_column(xc, embedding_size)

    # creates estimator
    estimator = tf.contrib.learn.DynamicRnnEstimator(
     config = run_config,
     problem_type = constants.ProblemType.CLASSIFICATION,
     prediction_type = PredictionType.SINGLE_VALUE,
     sequence_feature_columns = [xc],
     context_feature_columns = None,
     num_units = num_units,
     cell_type = cell_type, 
     optimizer = optimizer,
     learning_rate = learning_rate,
     num_classes = 2,
     dropout_keep_probabilities=drouput_keep_probabilities)

    train_input = get_input_fn(x_train, y_train, batch_size,
                               epochs=num_epochs,
                               max_length=max_length)
    test_input = get_input_fn(x_test, y_test, batch_size,
                              epochs=1,
                              max_length=max_length)

    return tf.contrib.learn.Experiment(
        estimator,
        train_input_fn=train_input,
        eval_input_fn=test_input,
    )

  return _experiment_fn


def main(unused_argv):
  # Loading the data
  # data from: https://keras.io/datasets/
  # Dataset of 25,000 movies reviews from IMDB, labeled by sentiment
  # (positive/negative).
  # Reviews have been preprocessed, and each review is encoded as a sequence
  # of word indexes (integers). 
  # For convenience, words are indexed by overall frequency in the dataset.

  NUM_WORDS = 1000  # using only the 1000 most common words

  print('Loading data...')

  (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=NUM_WORDS)

  print('size of the train dataset:', x_train.shape[0])
  print('size of the test dataset:', x_test.shape[0])

  # run experiment
  run_config = tf.contrib.learn.RunConfig(model_dir=FLAGS.model_dir)
  learn_runner.run(generate_experiment_fn(x_train, y_train,
                                          x_test, y_test,
                                          NUM_WORDS),
                   run_config=run_config)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)  # enable TensorFlow logs
  tf.app.run()

