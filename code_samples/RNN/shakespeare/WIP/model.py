#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# original code from: https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/blogs/timeseries
# modified by: Marianne Linhares, monteirom@google.com, May 2017

import tensorflow as tf
import shutil
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

SEQ_LEN = 10
DEFAULTS = [[0.0] for x in range(0, SEQ_LEN)]
BATCH_SIZE = 20
TIMESERIES_COL = 'rawdata'
N_OUTPUTS = 2  # in each sequence, 1-8 are features, and 9-10 is label
N_INPUTS = SEQ_LEN - N_OUTPUTS

# ------------- generate data --------------------
def create_time_series():
  freq = (np.random.random()*0.5) + 0.1  # 0.1 to 0.6
  ampl = np.random.random() + 0.5  # 0.5 to 1.5
  x = np.sin(np.arange(0,SEQ_LEN) * freq) * ampl
  return x

def to_csv(filename, N):
  with open(filename, 'w') as ofp:
    for lineno in range(0, N):
      seq = create_time_series()
      line = ",".join(map(str, seq))
      ofp.write(line + '\n')
 
to_csv('train.csv', 10000)  # 10000 sequences
to_csv('valid.csv',  50)

# -------- read data and convert to needed format -----------
def read_dataset(filename, mode=tf.estimator.ModeKeys.TRAIN):  
  def _input_fn():
    num_epochs = 100 if mode == tf.estimator.ModeKeys.TRAIN else 1

    # could be a path to one file or a file pattern.
    input_file_names = tf.train.match_filenames_once(filename)
    filename_queue = tf.train.string_input_producer(
        input_file_names, num_epochs=num_epochs, shuffle=True)

    reader = tf.TextLineReader()
    _, value = reader.read_up_to(filename_queue, num_records=BATCH_SIZE)

    value_column = tf.expand_dims(value, -1)
    print('readcsv={}'.format(value_column))
    
    # all_data is a list of tensors
    all_data = tf.decode_csv(value_column, record_defaults=DEFAULTS)  
    inputs = all_data[:len(all_data)-N_OUTPUTS]  # first few values
    label = all_data[len(all_data)-N_OUTPUTS : ] # last few values
    
    # from list of tensors to tensor with one more dimension
    inputs = tf.concat(inputs, axis=1)
    label = tf.concat(label, axis=1)
    print(inputs)
    print('inputs={}'.format(inputs))
    
    return {TIMESERIES_COL: inputs}, label   # dict of features, label
  return _input_fn

def get_train():
  return read_dataset('train.csv', mode=tf.estimator.ModeKeys.TRAIN)

def get_valid():
  return read_dataset('valid.csv', mode=tf.estimator.ModeKeys.EVAL)

def get_test():
  return read_dataset('test.csv', mode=tf.estimator.ModeKeys.EVAL)

# RNN Model
LSTM_SIZE = 3  # number of hidden layers in each of the LSTM cells

def simple_rnn(features, targets, mode, params):
  print('-' * 100)
  print(features[TIMESERIES_COL])
  # 0. Reformat input shape to become a sequence
  x = tf.split(features[TIMESERIES_COL], N_INPUTS, 1)
  #print 'x={}'.format(x)
    
  # 1. configure the RNN
  lstm_cell = rnn.BasicLSTMCell(LSTM_SIZE, forget_bias=1.0)
  outputs, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

  # slice to keep only the last cell of the RNN
  outputs = outputs[-1]
  #print 'last outputs={}'.format(outputs)
  
  # output is result of linear activation of last layer of RNN
  weight = tf.Variable(tf.random_normal([LSTM_SIZE, N_OUTPUTS]))
  bias = tf.Variable(tf.random_normal([N_OUTPUTS]))
  predictions = tf.matmul(outputs, weight) + bias
    
  # 2. Define the loss function for training/evaluation
  #print 'targets={}'.format(targets)
  #print 'preds={}'.format(predictions)
  loss = tf.losses.mean_squared_error(targets, predictions)
  eval_metric_ops = {
      "rmse": tf.metrics.root_mean_squared_error(targets, predictions)
  }
  
  # 3. Define the training operation/optimizer
  train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=0.01,
      optimizer="SGD")

  # 4. Create predictions
  predictions_dict = {"predicted": predictions}
  
  # 5. return ModelFnOps
  return tflearn.ModelFnOps(
      mode=mode,
      predictions=predictions_dict,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops)

def serving_input_fn():
    feature_placeholders = {
        TIMESERIES_COL: tf.placeholder(tf.float32, [None, N_INPUTS])
    }
  
    features = {
      key: tf.expand_dims(tensor, -1)
      for key, tensor in feature_placeholders.items()
    }

    return tflearn.utils.input_fn_utils.InputFnOps(
      features,
      None,
      feature_placeholders
    )

nn = tf.contrib.learn.Estimator(model_fn=simple_rnn)
# ---------- Training -------------
nn.fit(input_fn=get_train(), steps=10000)
ev = nn.evaluate(input_fn=get_valid())
print(ev)

# MANUAL TEST!!!!
to_csv('test.csv',  10)
print('-------------')
for e in nn.predict(input_fn=get_range()):
  print(e)

'''
for i in xrange(0, 5):
  sns.tsplot( create_time_series() )
'''

