#!/usr/bin/python3

"""
    Created by Talal Alrawajfeh on 09/18/2017
"""

import os
import time

import numpy as np

import mnist_data_reader
import nueral_network

BIASES_FILE_PATH = './out/biases.npy'
WEIGHTS_FILE_PATH = './out/weights.npy'

DEFAULT_NUMBER_OF_EPOCHS = 30
DEFAULT_BATCH_SIZE = 10
DEFAULT_LEARNING_RATE = 3.0
DEFAULT_NUMBER_OF_ROUNDS = 10


def main():
    neurons_per_layer = [mnist_data_reader.IMAGE_SIZE, 100, 10]
    network = nueral_network.NeuralNetwork(neurons_per_layer)

    if not os.path.isfile(WEIGHTS_FILE_PATH):
        start_time = time.time()
        for i in range(0, DEFAULT_NUMBER_OF_ROUNDS):
            training_data_reader = mnist_data_reader.MNISTDataReader(mnist_data_reader.TRAINING_DATA)
            for j in range(0, mnist_data_reader.NUMBER_OF_TRAINING_ITEMS // DEFAULT_BATCH_SIZE):
                print("Round: " + str(i + 1) + ", Batch: " + str(j + 1))
                batch = []
                for k in range(0, DEFAULT_BATCH_SIZE):
                    batch.append(training_data_reader.read_next_labeled_image())
                network.train(batch, DEFAULT_LEARNING_RATE, DEFAULT_BATCH_SIZE,
                              DEFAULT_NUMBER_OF_EPOCHS)
            training_data_reader.close()
        np.save(WEIGHTS_FILE_PATH, network.weights)
        np.save(BIASES_FILE_PATH, network.biases)
        duration = time.time() - start_time
        print("Duration: " + str(duration) + " seconds")
    else:
        network.weights = np.load(WEIGHTS_FILE_PATH)
        network.biases = np.load(BIASES_FILE_PATH)
        testing_data_reader = mnist_data_reader.MNISTDataReader(mnist_data_reader.TESTING_DATA)
        total = 0
        for i in range(0, mnist_data_reader.NUMBER_OF_TESTING_ITEMS):
            image, label = testing_data_reader.read_next_labeled_image()
            output = np.argmax(network.feed_forward(image))
            expected_output = np.argmax(label)
            if expected_output == output:
                total += 1
            print("Tests: " + str(total) + " of " + str(i + 1))


if __name__ == '__main__':
    main()
