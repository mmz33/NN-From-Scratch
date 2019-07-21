# Neural Network From Scratch

The purpose of this piece of code is to have a taste about what is really going on inside a Neural Network (NN). It is done for educational purposes and for fun :) Breifly, a NN is a function mapping from an input to some output depending on the task. In between lie what are called the "hidden layers" and there where all the magic is done. These layers have trainable parameters. The goal at the end is to minimize some loss function which is dependent on the network parameters. In order to find the optimal parameters, we use some kind of gradient descent optimizer (e.g stocastic gradient descent, Momentum, etc) with the help of the backpropagation algorithm.

## Datasets

Currently, there is a python script in `datasets` folder called `mnist.py` which basically downloads (if necessary) and prepares automatically this dataset for you. This can be extended later to support different datasets and of course you can implement your own dataset class and integrate it with the current code.

## Training

To train your model using the MNIST dataset, define it in `main.py` (as a task later this can be read from a python file for example or any other file kind) and just run: `python3 main.py --train`

You can also add more flags for the hyperparameters. For that, you can find more details by running `--help`. Later there can be a flag for choosing the dataset and based on that this dataset is used for training the model.

## Testing

To test your model, just run: `python3 main.py --test`

