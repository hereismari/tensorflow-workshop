import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators.dynamic_rnn_estimator import PredictionType

tf.logging.set_verbosity(tf.logging.INFO)

BATCH_SIZE = 128
SEQUENCE_LENGTH = 16

xc = tf.contrib.layers.real_valued_column("")
estimator = tf.contrib.learn.DynamicRnnEstimator(problem_type = constants.ProblemType.LINEAR_REGRESSION,
                                                 prediction_type = PredictionType.SINGLE_VALUE,
                                                 sequence_feature_columns = [xc],
                                                 context_feature_columns = None,
                                                 num_units = 5,
                                                 cell_type = 'lstm', 
                                                 optimizer = 'SGD',
                                                 learning_rate = 0.1)

def get_train_inputs():
  x = tf.random_uniform([BATCH_SIZE, SEQUENCE_LENGTH])
  y = tf.reduce_mean(x, axis=1)
  x = tf.expand_dims(x, axis=2)
  return {"": x}, y

def get_test_inputs():
  x = tf.random_uniform([1, SEQUENCE_LENGTH])
  y = tf.reduce_mean(x, axis=1)
  with tf.Session().as_default(): print(y.eval())
  x = tf.expand_dims(x, axis=2)
  return {"": x}, y


estimator.fit(input_fn=get_train_inputs, steps=2000)
p = estimator.predict(input_fn=get_test_inputs)
for e in p:
	print(e)
