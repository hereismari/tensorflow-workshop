import tensorflow as tf
print (tf.__version__) # tested with v1.2

# Keras
from tensorflow.contrib.keras.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# Model builder
from tensorflow.python.estimator import model_fn as model_fn_lib

# Define the model, using Keras
def get_model():
	def model_fn(features, labels, mode, params):

		conv1 = Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1))(features["x"])
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	   
		conv2 = Conv2D(64, (5, 5), activation='relu')(pool1)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		
		flat = Flatten()(pool2)
		dense = Dense(1024, activation='relu')(flat)

		logits = Dense(10, activation='softmax')(dense)

		loss = tf.losses.softmax_cross_entropy(
				onehot_labels=labels, logits=logits)
		
		train_op = tf.contrib.layers.optimize_loss(
				loss=loss,
				global_step=tf.contrib.framework.get_global_step(),
				learning_rate=params["learning_rate"],
				optimizer="SGD")
		
		predictions = {
			"classes": tf.argmax(input=logits, axis=1),
			"probabilities": tf.nn.softmax(logits)
		}
		
		eval_metric_ops = {
			"accuracy": tf.metrics.accuracy(
						 tf.argmax(input=logits, axis=1),
						 tf.argmax(input=labels, axis=1))
		}
		 
		return model_fn_lib.EstimatorSpec(
			mode=mode,
			predictions=predictions,
			loss=loss,
			train_op=train_op,
			eval_metric_ops=eval_metric_ops)
	return model_fn
