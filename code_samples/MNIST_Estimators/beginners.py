from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
print('Use TensorFlow version 1.2 or higher')
print('TensorFlow Version:', tf.__version__)

# Run an experiment
from tensorflow.contrib.learn.python.learn import learn_runner

# MNIST
from tensorflow.examples.tutorials.mnist import input_data

# Enable TensorFlow logs
tf.logging.set_verbosity(tf.logging.INFO)

# Parameters
BATCH_SIZE = 128
STEPS = 1000


def model_fn(features, labels, mode):
  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=features['x'], units=10)

  # Generate Predictions
  classes = tf.argmax(input=logits, axis=1)
  predictions = {
      'classes': classes,
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Return an EstimatorSpec object
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        learning_rate=1e-4,
        optimizer='Adam')

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                                      loss=loss, train_op=train_op)

  # Configure the accuracy metric for evaluation
  eval_metric_ops = {
      'accuracy': tf.metrics.accuracy(
          tf.argmax(input=logits, axis=1),
          tf.argmax(input=labels, axis=1))
  }

  return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                                    loss=loss, eval_metric_ops=eval_metric_ops)


# Import the MNIST dataset
mnist = input_data.read_data_sets('/tmp/MNIST/', one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

# Input functions
x_train_dict = {'x': x_train}
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x_train_dict, y_train, batch_size=BATCH_SIZE,
    shuffle=True, num_epochs=None,
    queue_capacity=1000, num_threads=1)

x_test_dict = {'x': x_test}
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x_test_dict, y_test, batch_size=BATCH_SIZE,
    shuffle=False, num_epochs=1)


# create experiment
def experiment_fn(run_config, hparams):
  del hparams  # unused arg
  # create estimator
  estimator = tf.estimator.Estimator(model_fn=model_fn,
                                     config=run_config)
  return tf.contrib.learn.Experiment(
      estimator,
      train_input_fn=train_input_fn,
      eval_input_fn=test_input_fn,
      train_steps=STEPS
  )

# run experiment
learn_runner.run(experiment_fn,
                 run_config=tf.contrib.learn.RunConfig(model_dir='beginners_output'))
