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


# Create multiple networks and train them to find the best one
# If the accuracy is not good enough, create a new network and start over
# Saves the current best network to networks/trained_network_<accuracy>.dat
def train_network_loop(training_data, test_data=None):
    # structure = [2500, 75, 30, 3]
    structure = [2500, 30, 3]
    print("Training network with {0} training examples and {1} validation chars".format(len(training_data), len(test_data)))
    print("Network structure: {0}".format(structure))

    found_best = False
    best_accuracy = 0
    best_net = None
    print("Starting to train networks to find the best one")
    while not found_best:
        net = Network(structure)
        accuracy = net.SGD(training_data, 40, 10, 3, test_data, True)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_net = net
            best_net.save("networks/trained_network_" + str(int(np.round(100 *  best_accuracy))) + ".dat")
            print("Found new best accuracy: {0}%".format(np.round(100 * best_accuracy)))
            print("Saved to trained_network.dat")
        if best_accuracy > 0.99:
            found_best = True

def train_network(training_data, test_data):
    print("Training network with {0} training examples and {1} validation chars".format(len(training_data), len(test_data)))
    print("Network structure: {0}".format([2500, 30, 3]))
    net = Network([2500, 30, 3])
    accuracy =net.SGD(training_data, 40, 10, 3, test_data, False)
    net.save("networks/trained_network_rnd.dat")
    print("Saved network with {0}% accuracy to networks/trained_network_rnd.dat".format(np.round(100 * accuracy)))

if __name__ == "__main__":
    training_data = read_training_data("data/training.dat")
    test_data = read_training_data("data/validation.dat")
    train_network(training_data, test_data)