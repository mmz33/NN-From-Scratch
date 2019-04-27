import numpy as np
from functools import partial


def gradient_descent(param, param_grad, lr):
    """
    Simple gradient descent

    :param param: parameter to be updated
    :param param_grad: gradient of the parameter
    :param lr: learning rate
    :return: updated parameter
    """
    return param - param_grad * lr


def train(model, loss, batch_size, epochs, lr, train_data, valid_data, valid_epoch=5):
    """Train model

    :param model: NN training model
    :param loss: loss module
    :param batch_size: the number of samples in the batch
    :param epochs: the number of training epochs
    :param lr: the value of the learning rate
    :param train_data: train DataSet object
    :param valid_data: valid DataSet object
    """

    model.init_network()
    update_func = partial(gradient_descent, lr=lr)  # other params will be filled later when called
    best_valid_loss = np.inf
    print('Start training...')
    for epoch in range(1, epochs+1):
        print('Start epoch {}/{}'.format(epoch, epochs))
        total_epoch_loss = 0.0
        n_batches = int(np.ceil(float(train_data.num_of_data)/batch_size))
        for batch_num in range(n_batches):

            # fetch data
            train_batch_data, train_batch_labels = train_data.next_batch(batch_size)

            # forward pass computation
            net_out = model.forward_prop(train_batch_data)

            # set the batch labels
            loss.set_targets(train_batch_labels)

            # compute the batch loss
            train_batch_loss = loss.forward_prop(net_out)
            batch_mean_loss = np.mean(train_batch_loss)
            total_epoch_loss += batch_mean_loss

            # print the average batch loss
            print('batch {} / loss: {}'.format(batch_num, batch_mean_loss))

            # backpropagate the loss and update the model params
            z = loss.back_prop(np.tile(1.0/len(train_batch_data), (len(train_batch_data), 1)))
            model.back_prop(z)
            model.update_network_params(update_func)

        print('Epoch loss:', total_epoch_loss/n_batches)

        if epoch % valid_epoch == 0:
            print('Start validation on epoch {}'.format(epoch))
            total_valid_loss = 0.0
            n_batches = int(np.ceil(float(valid_data.num_of_data)/batch_size))
            for batch_num in range(n_batches):
                valid_batch_data, valid_batch_labels = valid_data.next_batch(batch_size)
                valid_out = model.forward_prop(valid_batch_data)
                loss.set_targets(valid_batch_labels)
                valid_batch_loss = loss.forward_prop(valid_out)
                total_valid_loss += np.mean(valid_batch_loss)

            avg_loss = total_valid_loss/n_batches
            print('Validation Loss: {}'.format(avg_loss))
            if avg_loss >= best_valid_loss:
                print('Validation error is not improving... stopping')
                break  # early stopping
            else:
                best_valid_loss = avg_loss

    print('End training')
