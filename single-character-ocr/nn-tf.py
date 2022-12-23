import tensorflow as tf
import numpy as np

class Network(object):
    def __init__(self, sizes):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(2500)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(3, activation="softmax")
        ])

    def feedforward(self, a):
        return self.model.predict(a)

    def SGD(self, training_data, epochs, _, __, test_data=None, silent=False):
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(training_data, epochs=epochs, validation_data=test_data)
        model.evaluate(test_data)
        model.save("model.h5")

    # Evaluates the network for a give test_data set and returns the number of correct answers
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    # Saves the weights and biases of a network to a file
    def save(self, filename):
        self.model.save(filename)

    # Loads a network from a file
    @staticmethod
    def load(filename):
        net = Network([2500, 128, 3])
        net.model = tf.keras.models.load_model(filename)
        return net

