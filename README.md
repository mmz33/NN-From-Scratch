# Neural Network From Scratch

The purpose of this piece of code is to have a taste about what is really going on inside a Neural Network (NN). It is done for educational purposes and for fun :) Breifly, a NN is a function mapping an input to some output depending on the task. In between lie what are called the "hidden layers" and there where all the magic is done. These layers have trainable parameters. The goal at the end is to minimize some loss function which is dependent on the network parameters. In order to find the optimal parameters, we use some kind of gradient descent optimizer (e.g stocastic gradient descent, Momentum, etc) with the help of the backpropagation algorithm.

## Datasets

Currently, there is a python script in `datasets` folder called `mnist.py` which basically downloads (if necessary) and prepares automatically this dataset for you. This can be extended later to support different datasets and of course you can implement your own dataset class and integrate it with the current code.

## Components

- `main.py`: the main entry point.
- `config.py`: parse the json config file that contains the network and other (hyper)parameter.
- `engine.py`: backend engine that extracts content from the parsed json, construct the network layers, implements train and test functions.
- `nn_module.py`: represents a NN module such as a layer, activation function, loss function, etc.
- `model.py`: represents the NN model which is a stack of modules.
- `log.py`: represents a logger to control the output logs by using a log verbosity integer.
- `utils.py`: contains some helper functions.
- `tests.py`: contains functionality test functions.

## Dependencies

For dependencies, it is recommended to create a virtual enviroment and do `pip3 install -r requirements.txt`. But anyway the versions are not so important as the code uses basic methods. Moreover, `matplotlib` is only needed for this code in case you want to plot some MNIST images.

## Training

To train your model using the MNIST dataset, define the network and other parameters as a json file (see `configs/network1.json` for an example). For training, `task` should be set to `train` in the json config file. After that, you can just do:

`python3 main.py json_file`

The models will be saved in `model_file` which is defined in the json config file. They will be dumped as pickle files and can be loaded again for testing.

## Testing

For testing, just change the `task` in the json config to `test`. The results on the MNIST dataset with `configs/network1.json` config are:
```
Number of errors: 156/10000
Test accuracy: 98.44%
```
