from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras.python.keras.preprocessing import sequence
from tensorflow.contrib.learn.python.learn.estimators import rnn_common


def get_input_fn(x_in, y_in, batch_size,
                 shuffle=True, epochs=1, max_length=250,
                 batch_by_seq_len=False):

  def _length_bin(length, max_seq_len, length_step=10):
    # This function sets the sequence-length bin,
    # the returned value is the length of the longest
    # sequence allowed in the bin
    bin_id = (length // length_step + 1) * length_step
    return tf.cast(tf.minimum(bin_id, max_seq_len), tf.int64)

  def _make_batch(key, ds):
    # eliminate the extra padding
    key = tf.cast(key, tf.int32)
    ds = ds.map(lambda x, x_len, y: (x[:key], x_len, y))

    # convert the entire contents of the bin to a batch
    ds = ds.batch(batch_size)
    return ds

  def input_fn():
    # calculates the length of the sequences, where
    # length = min(actual_length, MAX_LENGT
    x_len = np.minimum(np.array([len(seq) for seq in x_in]),
                       max_length).astype('int32')

    # DynamicRNNEstimator uses `rnn_common.select_last_activations`:
    # https://goo.gl/L8jtfh
    # so we need add padding at the end of the sequence,
    # the default is the beginning of the sequence:
    # https://goo.gl/NVjJgT
    x_post_pad = sequence.pad_sequences(x_in, maxlen=max_length, padding='post')

    # creates the dataset from in memory data
    ds = tf.contrib.data.Dataset.from_tensor_slices((x_post_pad, x_len, y_in))

    # repeats the dataset `epochs` times.
    ds = ds.repeat(epochs)

    if shuffle:
      ds = ds.shuffle(buffer_size=10000)

    if batch_by_seq_len:
      # manually implement bucket by sequence length
      # the idea is to make batches with sequences of similar length
      # https://goo.gl/y67FQm
      ds = ds.group_by_window(
          key_func=lambda x, x_len, y: _length_bin(x_len, max_length),
          reduce_func=_make_batch,
          window_size=batch_size)
    else:
      ds = ds.batch(batch_size)

    # creates iterator
    x, x_len, y = ds.make_one_shot_iterator().get_next()

    dict_x = {'x': x, rnn_common.RNNKeys.SEQUENCE_LENGTH_KEY: x_len}
    return dict_x, y

  return input_fn

