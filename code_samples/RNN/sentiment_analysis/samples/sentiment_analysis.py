from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from input_functions import get_input_fn
from models import get_model_fn as CustomRNNEstimator
import tensorflow as tf
from tensorflow.contrib.keras.python.keras.datasets import imdb
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators.dynamic_rnn_estimator import PredictionType

print('TensorFlow version', tf.__version__)


parser = argparse.ArgumentParser()

parser.add_argument(
    '--is_canned_estimator', type=bool, default=True,
    help='If True the DynamicRNNEstimator will be loaded,'
         ' otherwise the CustomRNNEstimator will be loaded')

parser.add_argument(
    '--model_dir', type=str, default='sentiment_analysis_output',
    help='The directory where the model outputs should be stored')

parser.add_argument(
    '--batch_by_seq_len', type=bool, default=False,
    help='If True each bath will have sequences of similar length.'
         'This makes the model train faster')

parser.add_argument(
    '--max_len', type=int, default=250,
    help='Sentences will be truncated at max_len')

parser.add_argument(
    '--num_words', type=int, default=1000,
    help='Only num_words more frequent words will be used for testing')

parser.add_argument(
    '--train_batch_size', type=int, default=16,
    help='Batch size used for training')

parser.add_argument(
    '--eval_batch_size', type=int, default=16,
    help='Batch size used for evaluation')

parser.add_argument(
    '--embed_dim', type=int, default=30,
    help='Embedding dimension')

parser.add_argument(
    '--learning_rate', type=int, default=0.001,
    help='Learning rate')

parser.add_argument(
    '--num_epochs', type=int, default=10,
    help='Num epochs used for training (for evaluation is always 1)')

parser.add_argument(
    '--cell_type', type=str, default='lstm',
    help='RNN cell type')

parser.add_argument(
    '--optimizer', type=str, default='Adam',
    help='Optimizer used for training')

parser.add_argument(
    '--num_rnn_units', nargs='+', type=int, default=[256, 128],
    help='Size of the hidden state for each RNN cell')

parser.add_argument(
    '--num_dnn_units', nargs='+', type=int, default=[],
    help='Size of the hidden state for each RNN cell')

parser.add_argument(
    '--dropout_keep_probabilities', nargs='+', type=int,
    default=[0.9, 0.9, 0.9],
    help='Dropout probabilities to keep the cell. '
         'If provided the length should be num_rnn_units + 1')

parser.add_argument(
    '--num_classes', type=int, default=2,
    help='Number of output classes. '
         'For sentiment analysis is 2 (positive and negative)')

FLAGS = parser.parse_args()


# create experiment
def generate_experiment_fn(x_train, y_train, x_test, y_test):

  def _experiment_fn(run_config, hparams):
    del hparams  # unused arg

    # creates estimator
    if FLAGS.is_canned_estimator:
      xc = tf.contrib.layers.sparse_column_with_integerized_feature(
          'x',
          FLAGS.num_words)
      xc = tf.contrib.layers.embedding_column(xc, FLAGS.embed_dim)

      estimator = tf.contrib.learn.DynamicRnnEstimator(
          config=run_config,
          problem_type=constants.ProblemType.CLASSIFICATION,
          prediction_type=PredictionType.SINGLE_VALUE,
          sequence_feature_columns=[xc],
          context_feature_columns=None,
          num_units=FLAGS.num_rnn_units,
          cell_type=FLAGS.cell_type,
          optimizer=FLAGS.optimizer,
          learning_rate=FLAGS.learning_rate,
          num_classes=FLAGS.num_classes,
          dropout_keep_probabilities=FLAGS.dropout_keep_probabilities)

    else:
      model_fn = CustomRNNEstimator(
          rnn_cell_sizes=FLAGS.num_rnn_units,
          label_dimension=FLAGS.num_classes,
          num_words=FLAGS.num_words,
          dnn_layer_sizes=FLAGS.num_dnn_units,
          optimizer=FLAGS.optimizer,
          learning_rate=FLAGS.learning_rate,
          embed_dim=FLAGS.embed_dim)

      estimator = tf.estimator.Estimator(model_fn=model_fn,
                                         config=run_config)

    # input functions
    train_input = get_input_fn(x_train, y_train, FLAGS.train_batch_size,
                               epochs=FLAGS.num_epochs,
                               max_length=FLAGS.max_len,
                               batch_by_seq_len=FLAGS.batch_by_seq_len)
    test_input = get_input_fn(x_test, y_test, FLAGS.eval_batch_size,
                              epochs=1,
                              max_length=FLAGS.max_len)

    # returns Experiment
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

  print('Loading data...')

  (x_train, y_train), (x_test, y_test) = imdb.load_data(
      num_words=FLAGS.num_words)

  print('size of the train dataset:', x_train.shape[0])
  print('size of the test dataset:', x_test.shape[0])

  # run experiment
  run_config = tf.contrib.learn.RunConfig(model_dir=FLAGS.model_dir)
  learn_runner.run(generate_experiment_fn(x_train, y_train, x_test, y_test),
                   run_config=run_config)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)  # enable TensorFlow logs
  tf.app.run()

