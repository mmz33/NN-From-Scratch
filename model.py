from nn_module import NNModule
import inspect
import numpy as np
from collections import defaultdict


class NNModel:
    """
    This class represents the Neural Network Model which is a stack of modules (or layers)
    """

    def __init__(self, modules_list):
        """Build the model by creating the modules class instances

        :param modules_list: list of pair where each pair is a module class and it's args as dict
        """
        self.backprop_cache = defaultdict(None)  # used later for updated the parameters
        self.modules = []
        for (Module, args) in modules_list:
            assert inspect.isclass(Module), 'For now, only modules classes as input are supported'
            if not issubclass(Module, NNModule):
                raise Exception(str(Module.__class__.__name__) + ' is not supported')
            m = Module(**args)
            self.modules.append(m)

    def init_network(self):
        """Initialize the parameters of each module in the network"""
        for module in self.modules:
            module.init_params()

    def forward_prop(self, x):
        """
        Runs forward pass through all the modules of the network
        It will run from the input layer (first module) up to the output layer (last module)
        So, the output of one module is fed as an input to the next module and so on ...

        :param x: input
        :return output of the network
        """
        for m in self.modules:
            x = m.forward_prop(x)
        return x

    def back_prop(self, grad_out):
        """
        Runs backward pass through all the modules of the network
        It will run from the output layer (last module) up to the input layer (first module)
        So , the error of one module is propagated to the previous module (since we are moving backward)
        """
        for module in reversed(self.modules):
            self.backprop_cache[module] = np.array(grad_out)
            grad_out = module.back_prop(grad_out)

    def update_network_params(self, update_func):
        for module in self.modules:
            grad_params = module.get_params_grad(self.backprop_cache[module])
            if grad_params:
                module.params_update(update_func, grad_params)
