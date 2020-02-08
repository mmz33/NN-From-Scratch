import nn_module
import model
from datasets.mnist import read_datasets
from functools import partial
import numpy as np
import utils


class Engine:
    """
    Represents a backend engine that utilize the content of the json config to initiate the network, parameters,
    training, testing, etc
    """

    def __init__(self, config):
        """
        :param config: An instance of class Config.
        """
        self.config = config

    def init_from_config(self, is_train=True):
        """
        Extract some keys from the config

        :param is_train: A bool, if True then 'network' is initialized from the config
        """
        self.model_file = self.config.get_value('model_file')
        self.data_dir = self.config.get_value('data_dir')
        self.datasets = read_datasets(self.data_dir)  # contains train, valid, and test data
        self.valid_epoch = self.config.get_value('valid_epoch', 5)
        self.log_file = self.config.get_value('log_file')
        self.lr = self.config.get_value('lr', 0.7)
        self.num_epochs = self.config.get_value('num_epochs', 10)
        self.decay_rate = self.config.get_value('decay_rate', 0.3)
        self.start_epoch_decay = self.config.get_value('start_epoch_decay', 5)
        self.batch_size = self.config.get_value('batch_size', 100)
        self.loss = self.config.get_value('loss')
        self.loss_module = nn_module.get_module(self.loss)()

        # no need to init network for testing because a loaded pickle model is used
        if is_train:
            self.init_network_from_config()

    def init_network_from_config(self):
        """
        Loop over all the layers in the defined 'network' in json config and create all its layers
        """
        nn_modules = []
        net = self.config.get_value('network')
        for layer_name, layer_desc in net.items():
            assert isinstance(layer_desc, dict)
            layer_class = layer_desc['class']
            layer_module = nn_module.get_module(layer_class)
            del layer_desc['class']
            nn_modules.append((layer_module, layer_desc))
        self.net_model = model.NNModel(nn_modules)

    @staticmethod
    def gradient_descent(param, param_grad, lr):
        """
        Simple gradient descent

        :param param: parameter to be updated
        :param param_grad: gradient of the parameter
        :param lr: learning rate
        :return: updated parameter
        """
        return param - param_grad * lr

    def train(self):

        model = self.net_model
        model.init_network()
        update_func = partial(self.gradient_descent, lr=self.lr)  # other params will be filled later when called
        best_valid_loss = np.inf

        # used for learning rate scheduling depending on the epoch and decay rate
        curr_lr = self.lr
        decay_counter = 1

        train_data = self.datasets.train
        valid_data = self.datasets.valid

        print('Start training...')
        for epoch in range(1, self.num_epochs + 1):
            print('Start epoch {}/{}'.format(epoch, self.num_epochs))
            total_epoch_loss = 0.0
            n_batches = int(np.ceil(float(train_data.num_of_data) / self.batch_size))

            if epoch >= self.start_epoch_decay:
                lr_decay = self.decay_rate ** decay_counter
                curr_lr *= lr_decay
                decay_counter += 1
                update_func.keywords['lr'] = curr_lr

            for batch_num in range(n_batches):
                # fetch data
                train_batch_data, train_batch_labels = train_data.next_batch(self.batch_size)

                # forward pass computation
                net_out = model.forward_prop(train_batch_data)

                # set the batch labels
                self.loss_module.set_targets(train_batch_labels)

                # compute the batch loss
                train_batch_loss = self.loss_module.forward_prop(net_out)
                batch_mean_loss = np.mean(train_batch_loss)
                total_epoch_loss += batch_mean_loss

                # print the average batch loss
                print('batch {}/{} - loss: {}'.format(batch_num, n_batches, batch_mean_loss))

                # backpropagate the loss and update the model params
                z = self.loss_module.back_prop(np.tile(1.0 / len(train_batch_data), (len(train_batch_data), 1)))
                model.back_prop(z)
                model.update_network_params(update_func)

            print('Epoch {} loss: {}'.format(epoch, total_epoch_loss / n_batches))

            if epoch % self.valid_epoch == 0:
                print('Start validation on epoch {}'.format(epoch))
                total_valid_loss = 0.0
                n_batches = int(np.ceil(float(valid_data.num_of_data) / self.batch_size))
                for batch_num in range(n_batches):
                    valid_batch_data, valid_batch_labels = valid_data.next_batch(self.batch_size)
                    valid_out = model.forward_prop(valid_batch_data)
                    self.loss_module.set_targets(valid_batch_labels)
                    valid_batch_loss = self.loss_module.forward_prop(valid_out)
                    total_valid_loss += np.mean(valid_batch_loss)

                avg_loss = total_valid_loss / n_batches
                print('Validation loss: {}'.format(avg_loss))
                if avg_loss >= best_valid_loss:
                    print('Validation error is not improving... stopping')
                    break  # early stopping
                else:
                    best_valid_loss = avg_loss

        print('End training')

        if model and self.model_file:
            utils.save_model(self.model_file, model)

    def test_model(self):
        if self.model_file:
            model = utils.load_model(self.model_file)
        else:
            raise Exception('Can not find a trained model for testing')

        test_data = self.datasets.test
        num_of_data = test_data.num_of_data
        num_of_batches = int(np.ceil(float(num_of_data) / self.batch_size))
        pred_acc = 0  # prediction accuracy
        print('Start testing...')
        for batch_num in range(num_of_batches):
            test_batch_data, test_batch_labels = test_data.next_batch(self.batch_size)
            net_out = model.forward_prop(test_batch_data)
            preds = np.argmax(net_out, axis=1)  # network predictions
            batch_preds = np.isclose(preds, test_batch_labels)
            pred_acc += np.sum(batch_preds)

        print('Number of errors: {}/{}'.format(num_of_data - pred_acc, num_of_data))
        print('Error Rate: %.02f%%' % ((1.0 - pred_acc / num_of_data) * 100))
        print('Test accuracy: %.02f%%' % ((pred_acc / num_of_data) * 100))
