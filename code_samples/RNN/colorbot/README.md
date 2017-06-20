# Colorbot

Here is a draft implementation of ColorBot.

The idea came from [here](http://lewisandquark.tumblr.com/post/160776374467/new-paint-colors-invented-by-neural-network).
But unlike that blog post, we generate a color given a name, rather than generate a
name given a color.

The pre-trained model available at pretrained folder was trained on [this dataset]()
which we borrowed from [@andrewortman](https://github.com/andrewortman/colorbot/tree/master/data).

## About the model

Here's a diagram of the model used.

The model was trained in a way that given a sequence of lower case characteres
it tries to predict 3 float numbers that represent the normalized RGB values.
That are more likely for this sequence.

### Model

![]()

### Execution example

![]()


## About the dataset

The data available on this repo was taken from Wikipedia color dataset:

https://en.wikipedia.org/wiki/List_of_colors:_A-F
https://en.wikipedia.org/wiki/List_of_colors:_G-M
https://en.wikipedia.org/wiki/List_of_colors:_N-Z

The data was preprocessed and the format of the dataset that was actually used
can be seen at [insert link]()


For better results you can train your model in [this dataset]().
