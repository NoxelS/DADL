import numpy as np
from library import network
from library import layers
from library import misc

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

if __name__ == "__main__":
    training_data = read_training_data("data/training_data_bb1.dat")
    validation_data = read_training_data("data/validation_data_bb1.dat")

    print(f"Training network with {len(training_data)} training examples and {len(validation_data)} validation chars")
    net = network.Network(
         [
            layers.FullyConnectedLayer(2500, 30, misc.sigmoid, misc.sigmoid_prime),
            layers.FullyConnectedLayer(30, 3, misc.sigmoid, misc.sigmoid_prime),
         ]
    )

    accuracy = net.sgd(training_data, 7, 10, 3, validation_data)

    print(net.layers[1].weights)
