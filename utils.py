import pickle


def save_model(model_file, net_model):
    """Saves the network model to model file

    :param model_file: model file string path
    :param net_model: trained network model (learned params)
    """
    print('Saving model to file %s' % model_file)
    with open(model_file, 'wb') as f:
        pickle.dump(net_model, f)


def load_model(model_file):
    """Loads network model

    :param model_file: path to network model to load
    :return: loaded model
    """
    with open(model_file, 'rb') as f:
        return pickle.load(f)
