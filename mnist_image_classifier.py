#!/usr/bin/python3

"""
    Created by Talal Alrawajfeh on 09/22/2017
"""

import cv2

import numpy as np

import mnist_data_reader
import nueral_network

BIASES_FILE_PATH = './out/biases.npy'
WEIGHTS_FILE_PATH = './out/weights.npy'


def main():
    np.seterr(all='ignore')
    neurons_per_layer = [mnist_data_reader.IMAGE_SIZE, 100, 100, 100, 10]
    network = nueral_network.NeuralNetwork(neurons_per_layer)
    network.weights = np.load(WEIGHTS_FILE_PATH)
    network.biases = np.load(BIASES_FILE_PATH)
    image = cv2.imread('./resources/image.png', cv2.IMREAD_GRAYSCALE)
    print(np.argmax(network.feed_forward(np.reshape(image, (mnist_data_reader.IMAGE_SIZE, 1)))))


if __name__ == '__main__':
    main()
