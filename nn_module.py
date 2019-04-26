import abc
import numpy as np


class NNModule:
    """Abstract class representing an interface where every NN module has to implement"""

    __metaclass__ = abc.ABCMeta

    def init_params(self):
        """Initialize the module parameters"""
        pass

    def forward_prop(self, x):
        """Apply forward computation

        :param x: Input tensor. (B, n_in)
        :return output tensor of the next layer
        """
        return None

    def back_prop(self, grad_out):
        """Apply backward computation

        :param grad_out: output gradient from the l+1 layer. (1, n_out)
        :return gradient w.r.t input
        """
        return None

    def get_params_grad(self, grad_out):
        """Return the gradients of the module parameters

        :param grad_out: output gradient from the l+1 layer. (1, n_out)
        :return gradients w.r.t parameters
        """
        return None

    def params_update(self, update_func, grad_params):
        """Updates module parameters

        :param update_func: parameters update function such as gradient decent
        :param grad_params: list of parameters' gradients
        """
        pass


class FreeParamNNModule(NNModule):
    """Abstract class representing free (trainable) parameters modules"""

    __metaclass__ = abc.ABCMeta

    def init_params(self):
        # no parameters
        pass

    def get_params_grad(self, grad_out):
        # no params grads
        return None

    def params_update(self, update_func, grad_params):
        # no params update
        return None


class LossNNModule(NNModule):
    """Abstract class representing loss modules (e.g cross entropy loss, etc)"""

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.t = None

    def set_targets(self, t):
        """Set expected truth labels

        :param t: List of labels for each input in batch
        """
        self.t = t

    def init_params(self):
        # no parameters
        pass

    def get_params_grad(self, grad_out):
        # no params grads
        return None

    def params_update(self, update_func, grad_params):
        # no params update
        return None


class Linear(NNModule):
    """Represents a Linear layer that applies a linear projection"""

    def __init__(self, n_in, n_out):
        """
        :param n_in: number of input neurons
        :param n_out: number of output neurons
        :param W: weight matrix parameter. (n_in, n_out)
        :param b: bias parameter. (1, n_out)
        :param cache_input: used for backprop later (1, n_in)
        """

        self.n_in = n_in
        self.n_out = n_out
        self.W = None
        self.b = None
        self.cache_input = None

    def init_params(self):
        sigma = np.sqrt(2.0 / (self.n_in + self.n_out))
        self.W = np.random.normal(0, sigma, (self.n_in, self.n_out))
        self.b = np.zeros((1, self.n_out))

    def forward_prop(self, x):
        self.cache_input = x
        assert all(p is not None for p in [self.W, self.b]), 'W and b are not initialized!'
        return np.matmul(x, self.W) + self.b

    def back_prop(self, grad_out):
        return np.matmul(grad_out, np.transpose(self.W))

    def get_params_grad(self, grad_out):
        grad_W = np.matmul(np.transpose(self.cache_input), grad_out)
        grad_b = np.sum(grad_out, axis=0) if grad_out.ndim > 1 else grad_out
        return grad_W, grad_b

    def params_update(self, update_func, grad_params):
        self.W = update_func(self.W, grad_params[0])
        self.b = update_func(self.b, grad_params[1])


class Tanh(FreeParamNNModule):

    def __init__(self):
        self.cache_output = None

    def forward_prop(self, x):
        output = np.tanh(x)
        self.cache_output = np.array(output)
        return output

    def back_prop(self, grad_out):
        assert self.cache_output is not None, 'The output of tanh is not computed to apply backprop'
        return np.multiply(grad_out, 1.0 - self.cache_output * self.cache_output)


class ReLU(FreeParamNNModule):

    def __init__(self):
        self.cache_output = None

    def forward_prop(self, x):
        output = max(0, x)
        self.cache_output = output
        return output

    def back_prop(self, grad_out):
        assert self.cache_output is not None, 'The output of ReLU is not computed to apply backprop'
        return grad_out if self.cache_output > 0 else 0


class Softmax(FreeParamNNModule):
    """Represents a Softmax Layer

    Let x be represented as a vector [x_1, x_2, ..., x_n]
    then softmax(x) = [softmax(x_1), ..., softmax(x_n)]
    where softmax(x_i) = e^{x_i} / sum_{j}(e^{x_j})
    """

    def __init__(self):
        self.cache_output = None  # used for backprop

    def forward_prop(self, x):
        # here we can do the softmax trick for speed up and numerical stability
        # we can easily proof that softmax(x) = softmax(x - c)
        # so we can subtract the max value from x to reduce the computation

        max_value = np.max(x, axis=1)  # (B,)
        exps = np.exp((x.transpose() - max_value).transpose())  # use numpy broadcasting trick. (B, D)
        norm = np.sum(exps, axis=1)
        output = (exps.transpose() / norm).transpose()  # use broadcasting also here
        self.cache_output = np.array(output)
        return output

    def back_prop(self, grad_out):
        # z = grad_out * Jacobian(softmax)
        # where z_i = softmax(x_i)[grad_out_i - grad_out . softmax(x)^T]
        # we want to return z

        assert self.cache_output is not None, 'The output of softmax is not computed to apply backprop'

        if grad_out.ndim == 2:
            batch_size, n_out = grad_out.shape
        else:
            batch_size = 1
            n_out = len(grad_out)

        # compute the dot product (second term in z_i above)
        # note that v_s is (B, n_in) and not (B, 1) because later we want to subtract it with grad_out
        # Question: maybe it is better to set the 2nd dim to 1 and broadcast later?
        v_s = np.empty((batch_size, n_out))
        for i in range(batch_size):
            v_s[i, :] = np.dot(grad_out[i, :], self.cache_output[i, :])
        v_s = grad_out - v_s
        z = np.multiply(self.cache_output, v_s)
        return z


class CrossEntropyLoss(LossNNModule):
    """
    Represents the cross entropy loss function:

    CE(x) = -1 * sum_{i=1}^{N} { t_i * log[p(c_i|x)] }

    where x is an input vector, N is the number of classes, t_i is the truth label (0 or 1),
    and p(c_i | x) is the predicted (by NN) class for input x

    Note that softmax is coupled with cross entropy loss
    """

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.cache_input = None

    def forward_prop(self, x):
        """
        x is a predicted probability distribution over the classes
        it is basically the output of the softmax function

        t is a vector of labels for each input in the batch

        :param x: (B, n_in)
        """

        self.cache_input = np.array(x)
        batch_size = x.shape[0]
        loss = -1.0 * np.log(x[np.arange(batch_size), self.t])  # this will select a column from x and apply log to it
        return loss

    def back_prop(self, grad_out):
        """
        if we do the partial derivative of CE loss function w.r.t log(p(c_i|x)) we get:
            -1/p(c_i|x) if t_i = 1
            0           if t_i = 0

        So we just need to multiple the above jacobian of CE with grad_out to compute the output error
        """

        assert self.cache_input is not None, 'Cross entropy loss is not computed to apply backprop'
        assert self.cache_input.ndim == 2
        batch_size, n_in = self.cache_input.shape
        z = np.zeros((batch_size, n_in))
        z[np.arange(batch_size), self.t] = -1.0 / self.cache_input[np.arange(batch_size), self.t]
        np.multiply(grad_out, z, z)
        return z


class LogSoftmax(FreeParamNNModule):
    """
    LogSoftmax is simply log(Softmax)
    This can be faster and more numerically stable!

    log(softmax_i) = x_i - log(sum_j e^{x_j})

    """

    def __init__(self):
        self.cache_output = None

    def forward_prop(self, x):
        max_value = np.max(x, axis=1)  # (B,)
        x = (x.transpose() - max_value).transpose()
        exps = np.exp(x)
        norm = np.log(np.sum(exps, axis=1))
        output = (x.transpose() - norm).transpose()
        self.cache_output = np.array(output)
        return output

    def back_prop(self, grad_out):
        # z = grad_out * Jacobian(softmax)
        # where z_i = x_i - softmax(grad_out_i) * Sum(grad_out)
        # we want to return z

        assert self.cache_output is not None, 'The output of log softmax is not computed to apply backprop'

        grad_out_sum = np.sum(grad_out, axis=1, keepdims=True)  # (B, 1)
        exps = np.exp(self.cache_output)
        z = grad_out - np.multiply(grad_out_sum, exps)
        return z


class LogCrossEntropyLoss(LossNNModule):
    """
    Same as cross entropy loss but now with log (coupled with log softmax)
    """

    def __init__(self):
        super(LogCrossEntropyLoss, self).__init__()
        self.cache_input = None

    def forward_prop(self, x):
        self.cache_input = np.array(x)
        batch_size = x.shape[0]
        loss = -1.0 * x[np.arange(batch_size), self.t]  # x is already with log
        return loss

    def back_prop(self, grad_out):
        assert self.cache_input is not None, "The output of log ce is not computed to apply backprop"
        batch_size, n_in = self.cache_input
        z = np.zeros((batch_size, n_in))
        z[np.arange(batch_size), self.t] = -1
        return np.multiply(grad_out, z, z)
