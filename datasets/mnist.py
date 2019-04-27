import numpy as np
import matplotlib.pyplot as plt
import os.path
import urllib.request
import shutil
import gzip
import tempfile
import collections

"""This file represents functions and utils for downloading the MNIST dataset

The MNIST database is a huge database of handwritten digits that is commonly used for training, evaluating 
and comparing classifiers. It has a training set of 60,000 instances and a test set of 10,000 instances. 
Every instance is a 28 × 28 pixel gray-scale image.

The data files are compressed in bytes and looks as follows:

For images:
-----------

[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel


For labels:
-----------

[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  60000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
"""

# data urls
DEFAULT_URL = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

Datasets = collections.namedtuple('Datasets', ['train', 'valid', 'test'])


def maybe_download(source_url, file_name, work_dir):
    """Download (and unzip) a file from the MNIST dataset if not already done

    :param source_url: data url string
    :param file_name: data file name
    :param work_dir: data dir
    :return: data file path
    """
    file_path = os.path.join(work_dir, file_name)
    if os.path.exists(file_path):
        return file_path  # already done
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    _, zipped_filepath = tempfile.mkstemp(suffix='.gz')  # create a temporary file
    print('Downloading %s to %s' % (source_url, zipped_filepath))
    urllib.request.urlretrieve(source_url, zipped_filepath)  # download to temp file
    with gzip.open(zipped_filepath, 'rb') as f_in, \
            open(file_path, 'wb') as f_out:
         shutil.copyfileobj(f_in, f_out)
    os.remove(zipped_filepath)
    return file_path


def _read32(bytestream):
    """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
    dt = np.dtype(np.uint32).newbyteorder('>')  # big-endian memory
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
    """Extract the images

    :param f: A file object
    """
    with open(f, 'rb') as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                             (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        assert rows == 28 and cols == 28, 'Expected 28x28 images!'
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)  # interpreted as 1d array
        data = data.reshape(num_images, rows, cols, 1)
        return data


def extract_labels(f):
    """Extract labels

    :param f: A file object
    :return:
    """
    with open(f, 'rb') as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                             (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels


class DataSet(object):
    """
    This class constructs a dataset
    Here we have the option to reshape the images and normalize
    if images' data type is set to float32 then they are normalized to have values [0,1]
    """

    def __init__(self,
                 images,
                 labels,
                 dtype=np.float32,
                 reshape=True):
        """
        :param images: extracted images data
        :param labels: extracted labels data
        :param dtype: either uint8 or float32. If float32 then images are normalized
        :param reshape: If True, then images are reshaped to [N,W,H]
        """
        assert dtype in {np.uint8, np.float32}, 'dtype should be either uint8 or float32'
        assert images.shape[0] == labels.shape[0], \
            'images.shape[0] = {}, labels.shape[0] = {}'.format(images.shape[0], labels.shape[0])
        self._num_of_data = images.shape[0]
        if reshape:
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
        if dtype == np.float32:
            # we need to normalize from [0, 255] to [0.0, 1.0]
            images = images.astype(dtype)
            images /= 255.0
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def num_of_data(self):
        return self._num_of_data

    def plot_image_at_idx(self, idx):
        """Plot digit image

        :param idx: idx of the image in the dataset
        """
        assert 0 <= idx < self._num_of_data, 'image idx out of range'
        image = self._images[idx]
        assert image.ndim == 1 or image.ndim == 2
        if image.ndim == 1:
            image = image.reshape((28, 28))
        plt.title('Label is {}'.format(self._labels[idx]))
        plt.imshow(image, cmap='gray')
        plt.show()

    def next_batch(self, batch_size, shuffle=True):
        """
        Return the next `batch_size` examples from this data set.
        Shuffling is one way to avoid batches with same class as majority
        """
        start = self._index_in_epoch
        # shuffle the first epoch
        if self._epochs_completed and start == 0 and shuffle:
            perm0 = np.arange(self._num_of_data)
            np.random.shuffle(perm0)  # permutation of numbers between [0, num_of_data]
            # shuffle images and labels
            self._images = self._images[perm0]
            self._labels = self._labels[perm0]

        # go to next epoch
        if start + self._index_in_epoch > self._num_of_data:
            self._epochs_completed += 1

            # first collect the remaining data in case num of data is not divisible by the batch size
            # and use it for this epoch
            rem_data = self._num_of_data - start
            rem_images = self._images[start:self._num_of_data]
            rem_labels = self._labels[start:self._num_of_data]

            # shuffle data
            if shuffle:
                perm = np.arange(self._num_of_data)
                np.random.shuffle(perm)
                self._images = self._images[perm]
                self._labels = self._labels[perm]

            # start next epoch and add the rest data
            start = 0
            self._index_in_epoch = batch_size - rem_data
            end = self._index_in_epoch
            rest_images = self._images[start:end]
            rest_labels = self._labels[start:end]
            return np.concatenate((rem_images, rest_images)), np.concatenate((rem_labels, rest_labels))
        else:
            # return the data for this batch
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


# TODO: Maybe needs refactoring?
def read_datasets(train_dir, dtype=np.float32, reshape=True, validation_size=5000):
    # train images
    f = maybe_download(DEFAULT_URL + TRAIN_IMAGES, TRAIN_IMAGES, train_dir)
    train_images = extract_images(f)

    # train labels
    f = maybe_download(DEFAULT_URL + TRAIN_LABELS, TRAIN_LABELS, train_dir)
    train_labels = extract_labels(f)

    # test images
    f = maybe_download(DEFAULT_URL + TEST_IMAGES, TEST_IMAGES, train_dir)
    test_images = extract_images(f)

    # test labels
    f = maybe_download(DEFAULT_URL + TEST_LABELS, TEST_LABELS, train_dir)
    test_labels = extract_labels(f)

    assert 0 <= validation_size <= len(train_images), \
        'validation size should be between 0 and {}. Received: {}.'.format(validation_size, len(train_images))

    valid_images = train_images[:validation_size]
    valid_labels = train_images[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    params = dict(dtype=dtype, reshape=reshape)
    train = DataSet(train_images, train_labels, **params)
    valid = DataSet(valid_images, valid_labels, **params)
    test = DataSet(test_images, test_labels, **params)

    return Datasets(train=train, valid=valid, test=test)