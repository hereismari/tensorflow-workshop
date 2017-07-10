# [Draft] Sentiment Analysis Tutorial

In this tutorial we will show how to build a recurrent neural network for
[sentiment analysis](https://en.wikipedia.org/wiki/Sentiment_analysis) using
TensorFlow high level APIs.
Our task will consist of: given a sentence (sequence of words) classify
the sentence as positive or negative.

We will use the
[Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/),
which is a dataset for binary sentiment classification containing
25,000 reviews for training, and 25,000 reviews for testing with an
even number of positive and negative reviews.

In the end of this tutorial you will have learned more about:

  * [RNNs](https://www.tensorflow.org/tutorials/recurrent)
  * [Estimators](tensorflow.org/extend/estimators)
  * [Dataset API](https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/data)

## Introduction

Before get started, we recommend you to take a look at:

  * [Recurrent Neural Networks Tutorial](https://www.tensorflow.org/tutorials/recurrent)
  * [Word2Vec Tutorial](https://www.tensorflow.org/tutorials/word2vec)


## Describe the data in the details

* Option 1: use a bag of words from [Keras dataset](https://keras.io/datasets/)
  which can be loaded in memory and computed using a numpy input function or a
  dataset

* Option 2: use a [pretrained word2vec](https://github.com/adeshpande3/LSTM-Sentiment-Analysis).
  The data was generated for a oreilly tutorial. We can point to a link to the
  code and to the tensorflow tutorial about word2vec.
  We probably would have to use the dataset API for the input pipeline


We padded the data in a length of 250, which means that all the sequences now
have length of 250. If the sequence was bigger now it has only 250 elements
if was smaller we added zeros in the end.


## Show code and comment about it

WIP

## show the code training, final accuracy and predictions in practice

WIP
(show predictions in the test data)

## What's next?

In this tutorial we showed how to implement a recurrent neural network for
binary sentiment analysis using TensorFlow high level APIs.

We encourage you to run the code and see how the model performs for yourself.
The model parameters were not tuned, so a good exercise is just play with
the parameters and in order to have better results.
Try changing the learning rate, optimizer, hidden state size,
number of RNN cells, number of DNN layers, and so on.

Finally, the model presented above can be easilly changed to perform different
sequence classification or prediction tasks.
