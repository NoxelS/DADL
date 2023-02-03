from lib.utils import plot_missclassified_images, plot_activations
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
cnn = CNN('mnist', keep_checkpoints=True, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
          load_path='mnist.h5')

# Load model

cnn.fit(epochs=1000)
cnn.test()

# Get the missqualified images
plot_missclassified_images(
    x_test, y_test, cnn, path='figures/missclassified.png')

activation_imgaes = []
for i in range(10):
    # Find a test image that is calssidied as i
    for j in range(len(y_test)):
        if y_test[j] == i:
            # Reshape the image to a 4D tensor
            x_tmp = x_test[j]
            x_tmp = x_tmp.reshape(1, 28, 28, 1)
            x_tmp = x_tmp.astype('float32')  # Conv2d expects float32
            plot_activations(x_tmp, cnn, ['conv2d', 'max_pooling2d', 'conv2d_1', 'max_pooling2d_1',
                             'conv2d_2', 'max_pooling2d_2'], path=f'figures/activations_{i}.png')
            break
