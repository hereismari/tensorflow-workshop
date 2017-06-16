# Custom Estimator for MNIST using Keras and an input function.
# Based on:  * https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py 
# See also:
# * https://www.tensorflow.org/extend/estimators
# * https://www.tensorflow.org/get_started/mnist/beginners

# In[ ]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[ ]:

import tensorflow as tf
print (tf.__version__) # tested with v1.1


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

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('output_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_integer('num_gpus', 8,
                            """How many GPUs to use.""")

import time

def get_model(features):
	with tf.variable_scope("cnn_model") as scope:
		conv1 = Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1))(features["x"])
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  
		
		conv2 = Conv2D(64, (5, 5), activation='relu')(pool1)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		flat = Flatten()(pool2)
		dense = Dense(1024, activation='relu')(flat)

		logits = Dense(10)(dense)
		return logits
		
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
  for grad_and_vars in zip(*tower_grads):
    
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      
      # If gradient is None ignore it
      if g is not None:
        expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def tower_loss(features, targets, scope):
	logits = get_model(features)
	
	'''
	loss = tf.losses.softmax_cross_entropy(onehot_labels=targets, logits=logits)
	tf.add_to_collection('losses', loss)
	
	# Assemble all of the losses for the current tower only.
	losses = tf.get_collection('losses', scope)

	# Calculate the total loss for the current tower.
	total_loss = tf.add_n(losses, name='total_loss')

	return total_loss, logits
	'''
	cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=targets, logits=logits)
	loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
	return loss, logits

# Define the model, using Keras
def model_fn(features, targets, mode, params):
             
    # Create an optimizer that performs gradient descent.
    opt = tf.train.GradientDescentOptimizer(params["learning_rate"])
	
	# Calculate the gradients for each model tower.
    tower_grads = []
    # Storage avarege loss in a tensor
    average_loss_tensor = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('scope_gpu%d' % i) as scope:
            # Calculate the loss for one tower. This function
            # constructs the entire model but shares the
            # variables across all towers.
            loss, logits = tower_loss(features, targets, scope)

            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()
            
            # Storage avarege loss in a tensor
            # average_loss_tensor.append(loss)

            # Retain the summaries from the final tower.
            # summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            # Calculate the gradients for the batch of data on this tower
            grads = opt.compute_gradients(loss)

            # Keep track of the gradients across all towers.
            tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)
    
    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=tf.contrib.framework.get_global_step())
    
    	
    #avg_loss = tf.reduce_mean(tower_loss)
    #train_op = tf.contrib.layers.optimize_loss(loss=avg_loss,
    #                  	                        global_step=tf.contrib.framework.get_global_step(),
    #                                           learning_rate=params["learning_rate"],
    #                                           optimizer="SGD")

	# Loss
    # average_op = tf.reduce_mean(average_loss_tensor)
    average_op = loss 
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits)
    }
    
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(tf.argmax(input=logits, axis=1),
                                        tf.argmax(input=targets, axis=1))    
    }
			 
    return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=average_op,
                                   train_op=apply_gradient_op, eval_metric_ops=eval_metric_ops)

# In[ ]:

# Import the MNIST dataset
mnist = input_data.read_data_sets("/tmp/MNIST/", one_hot=True)

# using a smaller size to debug input fns
limit = 1000
x_train = np.reshape(mnist.train.images, (-1, 28, 28, 1))[:limit]
y_train = mnist.train.labels[:limit]
x_test = np.reshape(mnist.test.images, (-1, 28, 28, 1))[:limit]
y_test = mnist.test.labels[:limit]
    
# In[ ]:

# parameters
LEARNING_RATE = 0.01
BATCH_SIZE = 128
STEPS = 10000

# In[ ]:

# Input functions

x_train_dict = {'x': x_train }

train_input_fn = numpy_io.numpy_input_fn(
          x_train_dict, y_train, batch_size=BATCH_SIZE, 
           shuffle=False, num_epochs=None, 
            queue_capacity=1000, num_threads=1)

x_test_dict = {'x': x_test }
	
test_input_fn = numpy_io.numpy_input_fn(
          x_test_dict, y_test, batch_size=BATCH_SIZE, shuffle=False, num_epochs=1)

# In[ ]:

model_params = {"learning_rate": LEARNING_RATE}

# create estimator
estimator = tf.contrib.learn.Estimator(model_fn=model_fn, params=model_params)

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

    train_input = train_input_fn
    test_input = test_input_fn
    
    return tf.contrib.learn.Experiment(
        estimator,
        train_input_fn=train_input,
        eval_input_fn=test_input,
	    train_steps=STEPS
    )
  return _experiment_fn

# run experiment.
# Using TF v1.1 the second parameter doesn't seem to work,
# with TF v1.2 is possible to add a run_config parameter
# to select the folder you want to save the output files.
# TODO(monteirom): not crucial, but when you have the time
# try to figure it out if there's a way to choose the output
# folder with TF 1.1.0
start_time = time.time()
learn_runner.run(generate_experiment_fn(), FLAGS.output_dir)
print('--------------- seconds:', time.time() - start_time)
