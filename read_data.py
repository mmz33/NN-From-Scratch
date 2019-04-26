import numpy as np
import logging
import matplotlib.pyplot as plt


def read_data_files(data_path, labels_path):
    """Read images data

    :param data_path: path for images (train/test/valid)
    :param labels_path: path for labels (train/test/valid)
    :return: normalized images + corresponding labels (num_of_data, num_of_pixels)
    """
    logging.debug('Reading data...')
    data = np.genfromtxt(data_path, delimiter=' ', dtype=float)
    data /= 255.0  # normalize pixels
    labels = np.genfromtxt(labels_path, delimiter=' ', dtype=int)
    logging.debug('Data is read')
    return data, labels


def plot_mnist_image(image, label=None):
    """Plot MNIST image

    :param image: mnist image data pixels
    :param label: label [optional]
    :return:
    """
    pixels = np.array(image)
    assert pixels.ndim == 1
    # assuming the number of pixels is a perfect square
    img_dim = int(np.sqrt(pixels.shape[0]))
    pixels = pixels.reshape((img_dim, img_dim))
    if label:
        plt.title('Label is {}'.format(label))
    plt.imshow(pixels, cmap='gray')
    plt.show()
