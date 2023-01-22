import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return np.where(z > 0, 1, 0)

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)

def softmax_prime(z):
    return softmax(z) * (1 - softmax(z))