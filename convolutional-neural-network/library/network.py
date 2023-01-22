import numpy as np
import matplotlib.pyplot as plt
import random

class Network(object):

    def __init__(self, layers):
        self.layers = layers

    def feedforward(self, input_vector):
        output_vector = input_vector.copy()

        # Feed thourh layers
        for layer in self.layers:
            output_vector = layer.feedforward(output_vector)

        return output_vector

    def sgd(self, training_data, epochs, mini_batch_size, eta, validation_data, lmbda=0.0):
        accuracy = 0

        if validation_data : n_validation = len(validation_data)
        n_training = len(training_data)

        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n_training, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, mini_batch_size)

            if validation_data:
                accuracy = self.evaluate(validation_data) / n_validation
                print("Epoch {0}: {1} / {2} ({3}%)".format(epoch, self.evaluate(validation_data), n_validation, accuracy * 100))
            else:
                print("Epoch {0} complete".format(epoch))

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        # Update layers with backpropagation
        nabla_b = [np.zeros(layer.biases.shape) for layer in self.layers]
        nabla_w = [np.zeros(layer.weights.shape) for layer in self.layers]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.layers = [layer.update(eta, nb, nw, n) for layer, nb, nw in zip(self.layers, nabla_b, nabla_w)]


    def backprop(self, x, y):
        nabla_b = [np.zeros(layer.biases.shape) for layer in self.layers]
        nabla_w = [np.zeros(layer.weights.shape) for layer in self.layers]

        activation = x
        activations = [x]
        zs = []

        # Feedforward
        for layer in self.layers:
            z = layer.feedforward(activation)
            zs.append(z)
            activation = layer.activation_fn(z)
            activations.append(activation)

        # Backward pass
        delta = self.layers[-1].cost_derivative(activations[-1], y) * self.layers[-1].activation_fn_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, len(self.layers)):
            z = zs[-l]
            sp = self.layers[-l].activation_fn_prime(z)
            delta = np.dot(self.layers[-l+1].weights.transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def save(self, filename):
        # Save the neural network to the file "filename"
        pass

    def load(self, filename):
        # Load the neural network from the file "filename"
        pass