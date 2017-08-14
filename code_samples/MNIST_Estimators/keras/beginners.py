"""A Custom Estimator implementing linear regression for MNIST using Keras.

For reference:

* https://www.tensorflow.org/extend/estimators.
* https://www.tensorflow.org/get_started/mnist/beginners.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
print (tf.__version__)  # tested with v 1.2


import tensorflow.contrib.keras as K

# Model builder
from tensorflow.python.estimator import model_fn as model_fn_lib

# Input function
from tensorflow.python.estimator.inputs import numpy_io

# MNIST
from tensorflow.examples.tutorials.mnist import input_data

# Run an experiment
from tensorflow.contrib.learn.python.learn import learn_runner

# Enable TensorFlow logs
tf.logging.set_verbosity(tf.logging.INFO)


# Define the model, using Keras
def model_fn(features, labels, mode, params):
  """Model function for linear regressor."""
  logits = K.layers.Dense(10, input_dim=784)(features['x'])

  predictions = {
      'classes': tf.argmax(input=logits, axis=1),
      'probabilities': tf.nn.softmax(logits)
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return model_fn_lib.EstimatorSpec(mode=mode, predictions=predictions)

  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

  if mode == tf.estimator.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        learning_rate=params['learning_rate'],
        optimizer='Adam')

    return model_fn_lib.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op)

  eval_metric_ops = {
      'accuracy': tf.metrics.accuracy(
          tf.argmax(input=logits, axis=1),
          tf.argmax(input=labels, axis=1))
  }

  return model_fn_lib.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      eval_metric_ops=eval_metric_ops)

# Import the MNIST dataset
mnist = input_data.read_data_sets('/tmp/MNIST/', one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

# parameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
STEPS = 10000

model_params = {'learning_rate': LEARNING_RATE}

# Input functions
x_train_dict = {'x': x_train}
train_input_fn = numpy_io.numpy_input_fn(
    x_train_dict, y_train, batch_size=BATCH_SIZE,
    shuffle=True, num_epochs=None, queue_capacity=1000, num_threads=1)

x_test_dict = {'x': x_test}
test_input_fn = numpy_io.numpy_input_fn(
    x_test_dict, y_test, batch_size=BATCH_SIZE, shuffle=False, num_epochs=1)


# create experiment
def experiment_fn(run_config, hparams):
  # create estimator
  del hparams  # unused arg
  estimator = tf.estimator.Estimator(model_fn=model_fn,
                                     params=model_params,
                                     config=run_config)
  return tf.contrib.learn.Experiment(
      estimator,
      train_input_fn=train_input_fn,
      eval_input_fn=test_input_fn,
      train_steps=STEPS
  )

# run experiment
learn_runner.run(
    experiment_fn,
    run_config=tf.contrib.learn.RunConfig(model_dir='/tmp/beginners_mnist'))
