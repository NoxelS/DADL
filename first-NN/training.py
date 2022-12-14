import numpy as np
from nn import Network
# Read trainig.dat

def identifier_to_vector(identifier):
    if identifier == 'a':
        return np.array([1, 0, 0])
    elif identifier == 'b':
        return np.array([0, 1, 0])
    elif identifier == 'c':
        return np.array([0, 0, 1])


def read_training_data(filename):
    training_data = []
    with open(filename) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            entries = lines[i].split()
            identifier_vector = identifier_to_vector(entries.pop(0))
            greyscale_vector = np.array([float(x)/255.0 for x in entries])

            # Reshape to col vectors
            greyscale_vector = greyscale_vector.reshape((len(greyscale_vector), 1))
            identifier_vector = identifier_vector.reshape((len(identifier_vector), 1))

            training_data.append((greyscale_vector, identifier_vector))

            # print("Read training data: {0} / {1}".format(i, len(lines)))
    return training_data

def train_network(training_data, test_data=None):
    net = Network([2500, 30, 3])
    net.SGD(training_data, 30, 10, 3.0, test_data)
    net.save("trained_network.dat")

if __name__ == "__main__":
    training_data = read_training_data("training.dat")
    test_data = read_training_data("validation.dat")
    train_network(training_data, test_data)