#!/usr/bin/python3

"""
    Created by Talal Alrawajfeh on 09/18/2017
"""

import random

import numpy as np


class NeuralNetwork(object):
    def __init__(self, neurons_per_layer):
        self.neurons_per_layer = neurons_per_layer
        self.number_of_layers = len(neurons_per_layer)
        self.weights = [np.random.randn(x, y) for x, y in zip(neurons_per_layer[1:], neurons_per_layer[:-1])]
        self.biases = [np.random.randn(x, 1) for x in neurons_per_layer[1:]]

    def feed_forward(self, input_vector):
        output = input_vector
        for weight, bias in zip(self.weights, self.biases):
            output = sigmoid(np.dot(weight, output) + bias)
        return output

    def train(self, training_data, learning_rate, batch_size, epochs):
        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[j:j + batch_size] for j in range(0, n, batch_size)]
            for mini_batch in mini_batches:
                self.update_weights_and_biases(mini_batch, learning_rate)

    def back_propagate(self, input_vector, expected_output):
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        weighted_sums, activations = self.feed_forward_and_get_all_weighted_sums_and_activations(input_vector)
        kronecker_delta = cost_derivative(expected_output, activations[-1]) * sigmoid_derivative(weighted_sums[-1])
        nabla_w[-1] = np.dot(kronecker_delta, activations[-2].transpose())
        nabla_b[-1] = kronecker_delta
        for layer in range(2, self.number_of_layers):
            sd = sigmoid_derivative(activations[-layer])
            kronecker_delta = np.dot(self.weights[-layer + 1].transpose(), kronecker_delta) * sd
            nabla_w[-layer] = np.dot(kronecker_delta, activations[-layer - 1].transpose())
            nabla_b[-layer] = kronecker_delta
        return nabla_w, nabla_b

    def feed_forward_and_get_all_weighted_sums_and_activations(self, input_vector):
        previous_activation = input_vector
        activations = [previous_activation]
        weighted_sums = []
        for weight, bias in zip(self.weights, self.biases):
            weighted_sum = np.dot(weight, previous_activation) + bias
            weighted_sums.append(weighted_sum)
            previous_activation = sigmoid(weighted_sum)
            activations.append(previous_activation)
        return weighted_sums, activations

    def update_weights_and_biases(self, batch, learning_rate):
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        for inputs, output in batch:
            delta_nabla_w, delta_nabla_b = self.back_propagate(inputs, output)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            self.weights = [w - learning_rate / len(batch) * nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - learning_rate / len(batch) * nb for b, nb in zip(self.biases, nabla_b)]


def sigmoid(number):
    return 1.0 / (1.0 + np.exp(-number))


def cost_derivative(expected_output, output):
    return output - expected_output


def sigmoid_derivative(number):
    return sigmoid(number) * (1 - sigmoid(number))
