import tensorflow as tf

if tf.test.is_built_with_cuda():
  print("The installed version of TensorFlow includes GPU support.")
else:
  print("The installed version of TensorFlow does not include GPU support.")

