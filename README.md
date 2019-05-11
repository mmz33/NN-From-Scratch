# Neural Network From Scratch

The purpose of this peice of code is to have a taste about what is really going on inside a Neural Network (NN). It is done for educational purposes and fun :) Breifly, a NN is a function mapping from an input to some output depending on the task. In between relies what are called the "hidden layers" and there where all the majic is done. These layers have trainable parameters. The goal at the end is to minimize some loss function which is dependent on the network parameters. In order to find the optimal parameters, we use some kind of gradient descent optimizer (e.g stocastic gradient descent, Momentum, etc) with the help of the backpropagation algorithm.

### Datasets
---
Currently, there is a python script in `datasets` folder called `mnist.py` which basically downloads (if necessary) and prepares automatically this dataset for you. This can be expanded later to support different datasets and of course you can implement your own dataset class and integrate it with the current code.

### Training
---
To train your model, define it in `main.py` and just run: `python3 main.py --train`

You can also add more flags for the hyperparameters for example. For that, you can find more details by running `--help`

### Testing
---
To test your model, just run: `python3 main.py --test`

