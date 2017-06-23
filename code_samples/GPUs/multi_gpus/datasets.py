# Custom Estimator for MNIST using Keras and an input function.
# Based on:  * https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py 
# And: https://github.com/tfboyd/dlbench/blob/fcn5_updates/tools/tensorflow/fc/fcn5_mnist_multi_gpu.py
# See also:
# * https://www.tensorflow.org/extend/estimators
# * https://www.tensorflow.org/get_started/mnist/beginners

# In[ ]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[ ]:

import tensorflow as tf
print (tf.__version__) # tested with v1.2


# In[ ]:

# Keras
from tensorflow.contrib.keras.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# Estimators
from tensorflow.contrib import learn

# Model builder
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

# Input function
from tensorflow.python.estimator.inputs import numpy_io

# MNIST
from tensorflow.examples.tutorials.mnist import input_data

# Numpy
import numpy as np

# Enable TensorFlow logs
tf.logging.set_verbosity(tf.logging.INFO)

# Run an experiment
from tensorflow.contrib.learn.python.learn import learn_runner

import os
import functools
import operator

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('output_dir', 'proof_of_concept/workstation',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
                            
tf.flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay for convolutions.')

import time

def get_model(features, reuse, is_training):

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=features,
      filters=32,
      kernel_size=[5, 5],
      padding='same',
      activation=tf.nn.relu,
      reuse=reuse, name='conv1')

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                  pool_size=[2, 2], 
                                  strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding='same',
      activation=tf.nn.relu,
      reuse=reuse, name='conv2')

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2,
								  pool_size=[2, 2],
								  strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024,
                          activation=tf.nn.relu,
                          reuse=reuse, name='dense')

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=is_training)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10,
                           reuse=reuse, name='logits')
  return logits
  
def _tower_fn(is_training, weight_decay, features, labels, tower_losses,
              tower_gradvars, tower_preds, reuse):
  """Build computation tower for each device (CPU or GPU).
  Args:
    is_training: true if is for training graph.
    feature: a Tensor.
    label: a Tensor.
    tower_losses: a list to be appended with current tower's loss.
    tower_gradvars: a list to be appended with current tower's gradients.
    tower_preds: a list to be appended with current tower's predictions.
  """
  
  # with real models, when using GPUs: 
  # Computation requires channels first
  logits = get_model(features, reuse, is_training)
  tower_pred = {
      'classes': tf.argmax(input=logits, axis=1),
      'probabilities': tf.nn.softmax(logits)
  }
  tower_preds.append(tower_pred)

  tower_loss = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)
  tower_loss = tf.reduce_mean(tower_loss)
  tower_losses.append(tower_loss)

  model_params = tf.trainable_variables()
  tower_loss += weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in model_params])
  tower_losses.append(tower_loss)

  tower_grad = tf.gradients(tower_loss, model_params)
  tower_gradvars.append(zip(tower_grad, model_params))
		
def average_gradients(tower_grads):
	"""Calculate the average gradient for each shared variable across all towers.
	Note that this function provides a synchronization point across all towers.
	Args:
	tower_grads: List of lists of (gradient, variable) tuples. The outer list
	  is over individual gradients. The inner list is over the gradient
	  calculation for each tower.
	Returns:
	 List of pairs of (gradient, variable) where the gradient has been averaged
	 across all towers.
	"""
	average_grads = []
	for single_grads in zip(*tower_grads):
		grads = []
		for g, _ in single_grads:
			if g is not None:
				grads.append(g)
		grad = tf.add_n(grads)
		grad = tf.multiply(grad, 1.0/len(grads))
		v = single_grads[0][1]
		grad_and_var = (grad, v)
		average_grads.append(grad_and_var)
	return average_grads

def tower_loss(targets, logits):
	cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=targets, logits=logits)
	loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
	return loss

RUN = False

class ParamServerDeviceSetter(object):
  """Helper class to assign variables on the least loaded ps-device."""

  def __init__(self, worker_device, ps_devices):
    """Initializer for ParamServerDeviceSetter.
    Args:
      worker_device: the device to use for computer ops.
      ps_devices: a list of devices to use for Variable ops. Each variable is
      assigned to the least loaded device.
    """
    self.ps_devices = ps_devices
    self.worker_device = worker_device
    self.ps_sizes = [0] * len(self.ps_devices)

  def __call__(self, op):
    if op.device:
      return op.device
    if op.type not in ['Variable', 'VariableV2', 'VarHandleOp']:
      return self.worker_device

    device_index, _ = min(enumerate(self.ps_sizes), key=operator.itemgetter(1))
    device_name = self.ps_devices[device_index]
    var_size = op.outputs[0].get_shape().num_elements()
    self.ps_sizes[device_index] += var_size

    return device_name


def _create_device_setter(worker):
	"""Create device setter object."""
	gpus = ['/gpu:%d' % i for i in range(FLAGS.num_gpus)]
	return ParamServerDeviceSetter(worker, gpus)


def model_fn(features, labels, mode):
  """
  Support single host, one or more GPU training. Parameter distribution can be
  either one of the following scheme.
  1. CPU is the parameter server and manages gradient updates.
  2. Paramters are distributed evenly across all GPUs, and the first GPU
     manages gradient updates.
  Args:
    features: a list of tensors, one for each tower
    labels: a list of tensors, one for each tower
    mode: ModeKeys.TRAIN or EVAL
  Returns:
    A EstimatorSpec object.
  """

  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  num_gpus = FLAGS.num_gpus
  weight_decay = FLAGS.weight_decay

  tower_features = features
  tower_labels = labels
  tower_losses = []
  tower_gradvars = []
  tower_preds = []

  if num_gpus != 0:
    for i in range(num_gpus):
      worker = '/gpu:%d' % i
      device_setter = _create_device_setter(worker)
      reuse = (i != 0)
      with tf.variable_scope('resnet', reuse=reuse):
        with tf.name_scope('tower_%d' % i) as name_scope:
          with tf.device(device_setter):
            _tower_fn(is_training, weight_decay, tower_features[i],
                      tower_labels[i], tower_losses, tower_gradvars,
                      tower_preds, reuse)
            if i == 0:
              # Only trigger batch_norm moving mean and variance update from the
              # 1st tower. Ideally, we should grab the updates from all towers
              # but these stats accumulate extremely fast so we can ignore the
              # other stats from the other towers without significant detriment.
              update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                             name_scope)

  # Now compute global loss and gradients.
  gradvars = []
  
  # (For gpu-as-ps case, model params are distributed evenly across all gpus.)
  # It's the server that runs the ops to apply global gradient updates.
  with tf.device('/gpu:0'):
    with tf.name_scope('gradient_averaging'):
      loss = tf.reduce_mean(tower_losses)
      for zipped_gradvars in zip(*tower_gradvars):
        # Averaging one var's gradients computed from multiple towers
        var = zipped_gradvars[0][1]
        grads = [gv[0] for gv in zipped_gradvars]
        with tf.device(var.device):
          if len(grads) == 1:
            avg_grad = grads[0]
          else:
            avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
        gradvars.append((avg_grad, var))

	# Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

    # Create single grouped train op
    train_op = [
        optimizer.apply_gradients(
            gradvars, global_step=tf.train.get_global_step())
    ]
    train_op.extend(update_ops)
    train_op = tf.group(*train_op)

    predictions = {
        'classes':
            tf.concat([p['classes'] for p in tower_preds], axis=0),
        'probabilities':
            tf.concat([p['probabilities'] for p in tower_preds], axis=0)
    }
 
    stacked_labels = tf.concat(labels, axis=0)   
    metrics = {
        'accuracy': tf.metrics.accuracy(tf.argmax(stacked_labels, axis=1), predictions['classes'])
    }

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)

# In[ ]:

# Import the MNIST dataset
mnist = input_data.read_data_sets("/tmp/MNIST/", one_hot=True)

x_train = np.reshape(mnist.train.images, (-1, 28, 28, 1))#[:limit]
y_train = mnist.train.labels#[:limit]
x_test = np.reshape(mnist.test.images, (-1, 28, 28, 1))#[:limit]
y_test = mnist.test.labels#[:limit]
    
# In[ ]:

# parameters
BATCH_SIZE = 64 * FLAGS.num_gpus
STEPS = 2000


# Input functions
def input_fn(x, y, subset):
  """Create input graph for model.
  Args:
    subset: one of 'train', 'validate' and 'eval'.
  Returns:
    two lists of tensors for features and labels, each of num_gpus length.
  """
  is_training = (subset == 'train')
  batch_size = BATCH_SIZE
  
  num_shards = FLAGS.num_gpus
 
  with tf.device('/cpu:0'), tf.name_scope('batching'):
    dataset = tf.contrib.data.Dataset.from_tensor_slices([x, y])
    dataset = dataset.map(
        lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)),
        num_threads=2,
        output_buffer_size=batch_size)
 
    # Repeat infinitely
    dataset = dataset.repeat()
    if is_training:
      # Ensure that the capacity is sufficiently large to provide good random
      # shuffling
      dataset = dataset.shuffle(buffer_size= 4 * batch_size)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()
  
    # Note that passing num=batch_size is safe here, even though
    # dataset.batch(batch_size) can, in some cases, return fewer than batch_size
    # examples. This is because it does so only when repeating for a limited
    # number of epochs, but our dataset repeats forever.
    image_batch = tf.unstack(image_batch, num=batch_size, axis=0)
    label_batch = tf.unstack(label_batch, num=batch_size, axis=0)
    feature_shards = [[] for i in range(num_shards)]
    label_shards = [[] for i in range(num_shards)]
    for i in xrange(batch_size):
      idx = i % num_shards
      feature_shards[idx].append(image_batch[i])
      label_shards[idx].append(label_batch[i])
    feature_shards = [tf.parallel_stack(x) for x in feature_shards]
    label_shards = [tf.parallel_stack(x) for x in label_shards]
    return feature_shards, label_shards

def main(unused_argv):
  # The env variable is on deprecation path, default is set to off.
  os.environ['TF_SYNC_ON_FINISH'] = '0'

  if FLAGS.num_gpus <= 0:
    raise ValueError(
        'Invalid GPU count: \"num_gpus\" must be a positive integer.')
  if BATCH_SIZE % FLAGS.num_gpus != 0:
    raise ValueError('train_batch_size must be multiple of num_gpus.')

  config = tf.estimator.RunConfig()

  classifier = tf.estimator.Estimator(
      model_fn=model_fn, config=config, model_dir=FLAGS.output_dir)

  print('Starting to train...')
  start_time = time.time()
  classifier.train(
      input_fn=functools.partial(input_fn,
          x_train, y_train, subset='train'), steps=STEPS)
  print('--------------- seconds:', time.time() - start_time)

  print('Starting to evaluate...')
  eval_results = classifier.evaluate(
      input_fn=functools.partial(
          input_fn, x_test, y_test, subset='eval'), steps=100)
  print(eval_results)

if __name__ == '__main__':
  tf.app.run()
