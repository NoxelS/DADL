from lib.utils import *
from lib.cnn import CNN
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable tensorflow warnings

# Load and epochs the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension because the model expects a 4D input (batchsize x height x width x channels)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) # 8000 x 28 x 28 x 1
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1) # 2000 x 28 x 28 x 1

# Create a CNN instance
cnn = CNN(
    'mnist', 
    keep_checkpoints=True, 
    x_train=x_train, 
    y_train=y_train, 
    x_test=x_test, 
    y_test=y_test,
    load_path='models/mnist.h5'
)

# # Load model
# cnn.fit(epochs=5)
# cnn.test()

# Get the missqualified images and save them to a file
plot_missclassified_images(x_test, y_test, cnn, path='missclassified.png')

# Show kernels
kernel_visualisieren(kernel(cnn, 1))

# Show activation maps
shortened_model = shortening_model_to_layer(3, cnn.model)
activation_maps = activation_maps_up_to_last_layer(shortened_model, np.array([x_train[6]]))
show_activation_maps_of_desired_layer(activation_maps, 3)