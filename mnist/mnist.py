# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
Functions for downloading and reading MNIST data.

Modifications: Pierre Lelievre
"""

import os
import gzip
import numpy as np
from six.moves import urllib

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
N_DIGITS = 10
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def maybe_download(file_name, data_dir, verbose=False):
    """Download the data from Yann's website, unless it's already here."""
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    file_path = os.path.join(data_dir, file_name)
    if not os.path.exists(file_path):
        file_path, _ = urllib.request.urlretrieve(
            SOURCE_URL + file_name, file_path)
        stat_info = os.stat(file_path)
        if verbose:
            print(
                f'Successfully downloaded {file_name} {stat_info.st_size}'
                f' {bytes}.')
    return file_path


def _read32(byte_stream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(byte_stream.read(4), dtype=dt)[0]


def extract_images(file_name, verbose=False):
    """Extract the images into a 4D uint8 np array [index, y, x, depth]."""
    if verbose:
        print('Extracting', file_name)
    with gzip.open(file_name) as byte_stream:
        magic = _read32(byte_stream)
        if magic != 2051:
            raise ValueError(
                f'Invalid magic number {magic} in MNIST image file:'
                f' {file_name}')
        num_images = _read32(byte_stream)
        rows = _read32(byte_stream)
        cols = _read32(byte_stream)
        buf = byte_stream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols).astype(np.float32)
        return data / 255.0


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(file_name, one_hot=False, verbose=False):
    """Extract the labels into a 1D uint8 np array [index]."""
    if verbose:
        print('Extracting', file_name)
    with gzip.open(file_name) as byte_stream:
        magic = _read32(byte_stream)
        if magic != 2049:
            raise ValueError(
                f'Invalid magic number {magic} in MNIST image file:'
                f' {file_name}')
        num_items = _read32(byte_stream)
        buf = byte_stream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return dense_to_one_hot(labels)
        return labels


class MNIST:
    def __init__(self, data_dir=DATA_DIR, one_hot=False, verbose=False):
        self.data_dir = data_dir
        self.n_digits = N_DIGITS
        self.train_images = extract_images(maybe_download(
            TRAIN_IMAGES, self.data_dir, verbose), verbose)
        self.train_labels = extract_labels(maybe_download(
            TRAIN_LABELS, self.data_dir, verbose), one_hot, verbose)
        self.test_images = extract_images(maybe_download(
            TEST_IMAGES, self.data_dir, verbose), verbose)
        self.test_labels = extract_labels(maybe_download(
            TEST_LABELS, self.data_dir, verbose), one_hot, verbose)
