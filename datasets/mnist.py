import numpy as np
import logging
import matplotlib.pyplot as plt
import os.path
import urllib.request
import shutil
import gzip


def maybe_download(source_url, file_name, work_dir):
    """Downloads dataset only if it does not exist

    :param source_url:
    :param file_name:
    :param work_dir:
    :return:
    """
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    file_path = os.path.join(work_dir, file_name)
    if not os.path.exists(file_path):
        temp_file_name, _ = urllib.request.urlretrieve(source_url)
        shutil.copy(temp_file_name, file_path)
    return file_path

def extract_images():
    pass

def extract_labels():
    pass

def read_datasets(data_path, labels_path):
    """Read images data

    :param data_path: path for images (train/test/valid)
    :param labels_path: path for labels (train/test/valid)
    :return: normalized images + corresponding labels (num_of_data, num_of_pixels)
    """
    logging.info('Reading data...')

    TRAIN_IMAGES = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    TEST_IMAGES  = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    TEST_LABELS  = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

    with open(TRAIN_IMAGES, 'rb') as f:

    data = np.genfromtxt(data_path, delimiter=' ', dtype=float)
    data /= 255.0  # normalize pixels
    labels = np.genfromtxt(labels_path, delimiter=' ', dtype=int)
    logging.info('Data is read')
    return data, labels


def plot_image(image, label=None):
    """Plot digit image

    :param image: image data pixels
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
