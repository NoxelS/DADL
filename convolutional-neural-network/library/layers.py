import numpy as np
import matplotlib.pyplot as plt


class Layer(object):

    def __init__(self):
        raise NotImplementedError("Layer is an abstract class.")

    def feedforward(self, input_vector):
        raise NotImplementedError("Layer is an abstract class.")
    
    def update(self, eta, nb, nw, n):
        raise NotImplementedError("Layer is an abstract class.")
    
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

class FullyConnectedLayer(Layer):

    def __init__(self, input_size, output_size, activation_fn, activation_fn_prime):
        # Set weights and biases randomly
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)
        self.activation_fn = activation_fn
        self.activation_fn_prime = activation_fn_prime

    def feedforward(self, input_vector):
        return self.activation_fn(np.dot(self.weights, input_vector) + self.biases)

    def update(self, eta, nb, nw, n):
        self.weights = self.weights - (eta / n) * nw
        self.biases = self.biases - (eta / n) * nb
        return self

class ConvPoolLayer(Layer):

    def __init__(self, kernel_shape, image_shape, activation_fn, stride=1, padding=0, poolsize=(2,2)):
        # Initialize weights and biases
        pass

    def feedforward(self, input_vector):
        # Feed input through layers
        pass
