# coding: utf-8
'''A Custom Estimator using CNNS for MNIST using Keras.

For reference:

* https://www.tensorflow.org/extend/estimators.
* https://www.tensorflow.org/get_started/mnist/beginners.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
print (tf.__version__) # tested with v 1.2

# Keras
import tensorflow.contrib.keras as K

# Model builder
from tensorflow.python.estimator import model_fn as model_fn_lib

# Input function
from tensorflow.python.estimator.inputs import numpy_io

# MNIST
from tensorflow.examples.tutorials.mnist import input_data

# Enable TensorFlow logs
tf.logging.set_verbosity(tf.logging.INFO)

# Run an experiment
from tensorflow.contrib.learn.python.learn import learn_runner


# Define the model, using Keras
def model_fn(features, labels, mode, params):
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  x = tf.reshape(features['x'], shape=[-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = K.layers.Conv2D(32, (5, 5), activation='relu',
                          input_shape=(28, 28, 1))(x)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                strides=2,
                                padding='same')(conv1)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = K.layers.Conv2D(64, (5, 5), activation='relu')(pool1)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2.
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                strides=2,
                                padding='same')(conv2)

  # Flatten tensor into a batch of vectors.
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  flat = K.layers.Flatten()(pool2)

  # Dense Layer
  # Densely connected layer with 1024 neurons.
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = K.layers.Dense(1024, activation='relu')(flat)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = K.layers.Dense(10, activation='softmax')(dense)

  predictions = {
      'classes': tf.argmax(input=logits, axis=1),
      'probabilities': tf.nn.softmax(logits)
  }

  train_op = None
  eval_metric_ops = None

  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

  if mode == tf.estimator.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        learning_rate=params['learning_rate'],
        optimizer='Adam')

  if mode == tf.estimator.ModeKeys.EVAL:
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            tf.argmax(input=logits, axis=1),
            tf.argmax(input=labels, axis=1))
    }

  return model_fn_lib.EstimatorSpec(mode=mode, train_op=train_op,
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
STEPS = 1000

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
  del hparams  # unused arg
  # create estimator
  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
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
    run_config=tf.contrib.learn.RunConfig(model_dir='/tmp/pros_mnist')
)
