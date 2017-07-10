# https://www.tensorflow.org/extend/estimators
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# numpy
import numpy as np

# tensorflow
import tensorflow as tf

# keras
from tensorflow.contrib.keras.python.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from tensorflow.contrib.keras.python.keras import backend as K
K.set_learning_phase(1) #set learning phase

# input data
from tensorflow.examples.tutorials.mnist import input_data

# estimators
from tensorflow.contrib import learn

# estimator "builder"
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

import time

def time_usage(func):
  def wrapper(*args, **kwargs):
    beg_ts = time.time()
    retval = func(*args, **kwargs)
    end_ts = time.time()
    print("elapsed time: %f" % (end_ts - beg_ts))
    return retval
  return wrapper

# THE MODEL
def model_fn(features, targets, mode, params):
    """Model function for Estimator."""
    
    # 1. Configure the model via TensorFlow operations
    # First, build all the model, a good idea is using Keras or tf.layers
    # since these are high-level API's
    conv1 = Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1))(features)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
   
    conv2 = Conv2D(64, (5, 5), activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    flat = Flatten()(pool2)
    dense = Dense(1024, activation='relu')(flat)

    preds = Dense(10, activation='softmax')(dense)

    # 2. Define the loss function for training/evaluation
    loss = None
    train_op = None
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.PREDICT:
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=targets, logits=preds)

    # 3. Define the training operation/optimizer
    
    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=params["learning_rate"],
            optimizer="Adam",
        )

    # 4. Generate predictions
    predictions_dict = {
      "classes": tf.argmax(input=preds, axis=1),
      "probabilities": tf.nn.softmax(preds, name="softmax_tensor")
    }
    
    # 5. Define how you want to evaluate the model
    metrics = {
        "accuracy": tf.metrics.accuracy(tf.argmax(input=preds, axis=1), tf.argmax(input=targets, axis=1))
    }
    
    # 6. Return predictions/loss/train_op/eval_metric_ops in ModelFnOps object
    return model_fn_lib.ModelFnOps(
      mode=mode,
      predictions=predictions_dict,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)

@time_usage
def main():
    # Load datasets
    # mnist.train = 55,000 input data
    # mnist.test = 10,000 input data
    # mnist.validate = 5,000 input data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x_train = np.reshape(mnist.train.images, (-1, 28, 28, 1))
    y_train = mnist.train.labels
    x_test = np.reshape(mnist.test.images, (-1, 28, 28, 1))
    y_test = mnist.test.labels
    
    # PARAMETERS
    LEARNING_RATE = 0.001
    BATCH_SIZE = 16
    STEPS = 10000

    # Set model params
    model_params = {"learning_rate": LEARNING_RATE}

    # Instantiate Estimator
    nn = tf.contrib.learn.Estimator(model_fn=model_fn, params=model_params)

    # Fit
    print('-' * 40)
    print("Training")
    print('-' * 40)
    nn.fit(x=x_train, y=y_train, steps=STEPS,  batch_size=BATCH_SIZE)

    # Score accuracy
    
    print('-' * 40)
    print("Testing")
    print('-' * 40)
    ev = nn.evaluate(x=x_test, y=y_test, steps=1)
    print("Loss: %s" % ev["loss"])
    print("Accuracy: %f" % ev["accuracy"])
    
if __name__ == "__main__":
    main()

