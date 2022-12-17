import numpy as np
from nn import Network

# Translate the identifier to a vector, e.g. 'a' -> [1, 0, 0]
# You can introduce your  own identifiers here
def identifier_to_vector(identifier):
	vector = np.zeros(3)
	vector[ord(identifier)-ord('a')] = 1
	return vector

# Read the training data from the file. The file should have the following format:
#   <identifier> <pixel 1> <pixel 2> ... <pixel 2500>
def read_training_data(path):
    training_data = []
    with open(path) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            entries = lines[i].split()
            identifier_vector = identifier_to_vector(entries.pop(0))
            greyscale_vector = np.array([float(x)/255.0 for x in entries])

            # Reshape to col vectors
            greyscale_vector = greyscale_vector.reshape((len(greyscale_vector), 1))
            identifier_vector = identifier_vector.reshape((len(identifier_vector), 1))

            training_data.append((greyscale_vector, identifier_vector))

    return training_data


# Create multiple networks and train them to find the best one
# If the accuracy is not good enough, create a new network and start over
# Saves the current best network to networks/trained_network_<accuracy>.dat
def train_network_loop(training_data, structure, test_data=None):
    print(f"Training network with {len(training_data)} training examples and {len(test_data)} validation chars")
    print(f"Network structure: {structure}")

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
            print(f"Found new best accuracy: {np.round(100 * best_accuracy)}%")
        
        # Stop if we have a good enough accuracy
        if best_accuracy > 0.99:
            found_best = True

# Train a single network and save it to a file
def train_network(training_data, test_data, structure, save_path):
    print(f"Training network with {len(training_data)} training examples and {len(test_data)} validation chars")
    print(f"Network structure: {structure}")
    net = Network(structure)
    accuracy = net.SGD(training_data, 40, 10, 3, test_data, False)
    net.save(save_path)
    print(f"Saved network with {np.round(100 * accuracy)}% accuracy to {save_path}")

if __name__ == "__main__":
    # Load the training data from file
    training_data = read_training_data("data/training.dat") # Change this to your training data
    test_data = read_training_data("data/validation.dat") # Change this to your validation data

    # Train the network with either train_network_loop or train_network
    # train_network_loop(training_data, [2500, 30, 3], test_data)
    train_network(training_data, test_data, [2500, 30, 3], "networks/trained_network.dat")

