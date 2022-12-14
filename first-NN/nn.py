import numpy as np
import random
import json
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg') # no UI backend

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigprime(z):
    return sigmoid(z)*(1-sigmoid(z))

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes

        # Biases
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # Weight for layer i and j
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    # Return the output of the network if "a" is input
    def feedforward(self, a):
        A = a.copy() # Copy the input so we don't change it
        for b, w in zip(self.biases, self.weights): # Layer for layer
            A = sigmoid(np.dot(w, A)+b)
        return A

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        plotX = []
        plotY = []

        if test_data : n_test = len(test_data)
        n = len(training_data)
        best_eval = 0
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data :
                eval= self.evaluate(test_data)
                if eval/n_test > best_eval:
                    best_eval = eval/n_test 
                plotX.append(j)
                plotY.append(eval)
                print("Epoch {0}: {1} / {2}".format(j, eval, n_test))
            else:
                print("Epoch {0} complete".format(j))
        if test_data :
            plt.plot(plotX, np.multiply(plotY,1/n_test))
            plt.xlabel("Epoch")
            plt.ylabel("Correct")
            plt.title("Training for {0} epochs [{1}, {2}, {3}]".format(epochs, self.sizes[0], self.sizes[1], self.sizes[2]))
            plt.savefig("training.png")

        self.save("trained_network_best_" + str(np.round(100 * best_eval, 2)) + ".dat")

        try:
            plt.show()
        except:
            pass

    def update_mini_batch(self, mini_batch, eta):
        # nabla_b and nabla_w are the gradient of the cost function
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        # nabla_b and nabla_w are the gradient of the cost function ?
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

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases]}
        with open(filename, 'w') as f:
            json.dump(data, f)
        f.close()

    @staticmethod
    def load(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        f.close()
        net = Network(data["sizes"])
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]
        return net

if(__name__ == '__main__'):
    net = Network([2, 3, 1])