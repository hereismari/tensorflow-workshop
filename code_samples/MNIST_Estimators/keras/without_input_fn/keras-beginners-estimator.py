# https://www.tensorflow.org/extend/estimators
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# tensorflow
import tensorflow as tf

# keras
from tensorflow.contrib.keras.python.keras.layers import Dense

# input data
from tensorflow.examples.tutorials.mnist import input_data

# estimators
from tensorflow.contrib import learn

# estimator "builder"
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

# THE MODEL
def model_fn(features, targets, mode, params):
    """Model function for Estimator."""
    
    # 1. Configure the model via TensorFlow operations
    # First, build all the model, a good idea is using Keras or tf.layers
    # since these are high-level API's
    preds = Dense(10, input_dim=784)(features)

    # 2. Define the loss function for training/evaluation
    
    loss = None
    train_op = None
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=targets, logits=preds)

    # 3. Define the training operation/optimizer
    
    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=params["learning_rate"],
            optimizer="SGD",
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

def main():
    # Load datasets
    # mnist.train = 55,000 input data
    # mnist.test = 10,000 input data
    # mnist.validate = 5,000 input data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x_train = mnist.train.images
    y_train = mnist.train.labels
    x_test = mnist.test.images
    y_test = mnist.test.labels
    
    # PARAMETERS
    LEARNING_RATE = 0.01
    BATCH_SIZE = 128
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
    print("Accuracy: %f" % ev["accuracy"] * 100)
    
if __name__ == "__main__":
    main()

