# Colorbot

**Special thanks to [@andrewortman](https://github.com/andrewortman/colorbot/) that has a repo with a good implementation of colorbot using TensorFlow, and gave us the idea to make a workshop about it!**

Here is a draft implementation of ColorBot.

The idea came from [here](http://lewisandquark.tumblr.com/post/160776374467/new-paint-colors-invented-by-neural-network).
But unlike that blog post, we generate a color given a name, rather than generate a
name given a color.

The pre-trained model available at pretrained folder was trained on [this dataset (MISSING LINK)]()
which was preprocessed by [@andrewortman](https://github.com/andrewortman/colorbot/).

![](https://github.com/mari-linhares/tensorflow-workshop/blob/master/code_samples/RNN/colorbot/imgs/model_gif.gif)

See ColorBot in action running: *python play_colorbot.py*


## About the model

Here's a diagram of the model used.

The model was trained in a way that given a sequence of lower case characteres
it tries to predict 3 float numbers that represent the normalized RGB values.
That are more likely for this sequence.

### Model

![](https://github.com/mari-linhares/tensorflow-workshop/blob/master/code_samples/RNN/colorbot/imgs/colorbot_model.png)

### Execution example

![](https://github.com/mari-linhares/tensorflow-workshop/blob/master/code_samples/RNN/colorbot/imgs/colorbot_execution.png)


## About the dataset

The data available on this repo was taken from Wikipedia color dataset:

https://en.wikipedia.org/wiki/List_of_colors:_A-F  
https://en.wikipedia.org/wiki/List_of_colors:_G-M  
https://en.wikipedia.org/wiki/List_of_colors:_N-Z

The data was preprocessed and the format of the dataset that was actually used
can be seen [here](https://github.com/mari-linhares/tensorflow-workshop/blob/master/code_samples/RNN/colorbot/data/test.csv)

For better results you can train your model in [this dataset (MISSING LINK) ]().
