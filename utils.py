import pickle
import os


def save_model(model_file, net_model):
    """
    Saves the network model to model file

    :param model_file: A string, model file string path
    :param net_model: A model, trained network model (learned params)
    """
    print('Saving model to %s' % model_file)
    if '/' in model_file:
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
    with open(model_file, 'wb') as f:
        pickle.dump(net_model, f)


def load_model(model_file):
    """
    Loads network model

    :param model_file: A string, path to network model to load
    :return: loaded model
    """
    with open(model_file, 'rb') as f:
        return pickle.load(f)
