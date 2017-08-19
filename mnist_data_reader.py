#!/usr/bin/python3

"""
    Created by Talal Alrawajfeh on 09/17/2017
"""

import gzip
import numpy as np

IMAGE_PIXEL_ROWS = 28
IMAGE_PIXEL_COLUMNS = 28
IMAGE_SIZE = IMAGE_PIXEL_ROWS * IMAGE_PIXEL_COLUMNS
IMAGES_OFFSET = 16

LABEL_SIZE = 1
LABELS_OFFSET = 8

TRAINING_IMAGES_FILE = './resources/train-images-idx3-ubyte.gz'
TRAINING_LABELS_FILE = './resources/train-labels-idx1-ubyte.gz'
TESTING_IMAGES_FILE = './resources/t10k-images-idx3-ubyte.gz'
TESTING_LABELS_FILE = './resources/t10k-labels-idx1-ubyte.gz'

NUMBER_OF_TRAINING_ITEMS = 60000
NUMBER_OF_TESTING_ITEMS = 10000

TRAINING_DATA = 1
TESTING_DATA = 2


def open_gzip(file_name):
    return gzip.open(file_name, 'rb')


def label_to_vector(l):
    vector = np.zeros((10, 1))
    vector[l] = 1.0
    return vector


class InputError(Exception):
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class MNISTDataReader:
    def __init__(self, data_type=TRAINING_DATA):
        self.data_type = data_type
        if data_type == TRAINING_DATA:
            self.images_file = open_gzip(TRAINING_IMAGES_FILE)
            self.labels_file = open_gzip(TRAINING_LABELS_FILE)
        elif data_type == TESTING_DATA:
            self.images_file = open_gzip(TESTING_IMAGES_FILE)
            self.labels_file = open_gzip(TESTING_LABELS_FILE)
        else:
            raise InputError
        self.images_file.seek(IMAGES_OFFSET)
        self.labels_file.seek(LABELS_OFFSET)

    def read_next_labeled_image(self):
        pixels = np.array(list(map(lambda x: int(x), self.images_file.read(IMAGE_SIZE))))
        l = self.labels_file.read(LABEL_SIZE)
        return np.reshape(pixels, (IMAGE_SIZE, 1)), label_to_vector(int.from_bytes(l, 'big'))

    def close(self):
        self.images_file.close()
        self.labels_file.close()
