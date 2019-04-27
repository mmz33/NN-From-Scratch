import numpy as np


def test_model(model, batch_size, test_data):
    """Test model

    :param model: trained nn model to be used for testing
    :param batch_size: the number of samples in the batch
    :param test_data: test dataset (images + labels)
    """

    num_of_data = test_data.num_of_data
    num_of_batches = int(np.ceil(float(num_of_data)/batch_size))
    pred_acc = 0  # prediction accuracy
    print('Start testing...')
    for batch_num in range(num_of_batches):
        test_batch_data, test_batch_labels = test_data.next_batch(batch_size)
        net_out = model.forward_prop(test_batch_data)
        preds = np.argmax(net_out, axis=1)  # network predictions
        batch_preds = np.isclose(preds, test_batch_labels)
        pred_acc += np.sum(batch_preds)

    print('Number of errors: {}/{}'.format(num_of_data - pred_acc, num_of_data))
    print('Error Rate: %.02f%%' % ((1.0 - pred_acc/num_of_data) * 100))
    print('Test accuracy: %.02f%%' % ((pred_acc/num_of_data) * 100))
