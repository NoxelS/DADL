import numpy as np
import random
import json
from matplotlib import pyplot as plt

import matplotlib
# Force matplotlib to not use any Xwindows backend because my wsl is broken
matplotlib.use('Agg')

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigprime(z):
    return sigmoid(z)*(1-sigmoid(z))

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        A = a.copy() # Copy the input so we make sure we don't change it
        for b, w in zip(self.biases, self.weights): # Layer for layer calculating the output
            A = sigmoid(np.dot(w, A)+b)
        return A

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None, silent=False):
        plotX, plotY = [], [] # Used for plotting loss
        accuracy = 0 # Used for aborting training if accuracy is too low

        if test_data : n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            # Shuffle the training data so we don't get stuck in a local minimum
            # and split the trainign data into mini batches
            random.shuffle(training_data) 
            mini_batches = [ 
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            # Update the weights and biases for each mini batch
            for mini_batch in mini_batches: 
                self.update_mini_batch(mini_batch, eta)

            # Evaluate the network after each epoch and print the current accuracy for the test data
            if test_data :
                last_eval = self.evaluate(test_data)
                plotX.append(j)
                plotY.append(last_eval)
                accuracy = last_eval/n_test
                if not silent:
                    print("Epoch {0}: {1} / {2} ({3}%)".format(j, last_eval, n_test, np.round(100 * last_eval/n_test, 2)))
            else:
                if not silent:
                    print("Epoch {0} complete".format(j))
            
            # Abort training if accuracy is too low
            if j > 3 and accuracy < 0.9:
                if not silent:
                    print("Accuracy is too low, aborting training")
                return accuracy

        # Plot the accuracy over epochs and save the plot
        if test_data :
            plt.plot(plotX, np.multiply(plotY,1/n_test))
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Training for {0} epochs [{1}, {2}, {3}]".format(epochs, self.sizes[0], self.sizes[1], self.sizes[2]))
            plt.savefig("training.png")
            if not silent:
                print("Training complete, saved plot as training.png")

        return accuracy

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x] # List to store all the activations, layer by layer
        zs = [] # List to store all the z vectors, layer by layer

        # Feedforward
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * sigprime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # L-1 th layer to the second layer
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigprime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    # Evaluates the network for a give test_data set and returns the number of correct answers
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

    # Saves the weights and biases of a network to a file
    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases]}
        with open(filename, 'w') as f:
            json.dump(data, f)
        f.close()

    # Loads a network from a file
    @staticmethod
    def load(filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            f.close()
            net = Network(data["sizes"])
            net.weights = [np.array(w) for w in data["weights"]]
            net.biases = [np.array(b) for b in data["biases"]]
        except:
            print("Error loading network. Is your network saved as a json file?")
            return None
        return net