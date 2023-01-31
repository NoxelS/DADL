import tensorflow as tf
from keras import datasets
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images, test_images = train_images / 255.0 , test_images / 255.0

cnn = tf.keras.models.load_model("/home/max/Desktop/DADL/cnn/models/aug_cnn_model.h5")

#layer werden nach Zählweise im skript durchgezählt
def kernel(model, layer):
    kernels, biases = model.layers[layer-1].get_weights()
    return kernels

def kernel_visualisieren(kernels):
    n_kernels = kernels.shape[3]

    #Normierung der weights innerhalb der kernel
    kernels = (kernels - kernels.min()) / (kernels.max() - kernels.min())

    #Anzeigen aller kernels in einem Fenster
    for i in range(n_kernels):
        ax = plt.subplot(1, n_kernels, i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(kernels[:,:,0,i], cmap='gray')
    plt.show()

kernel_visualisieren(kernel(cnn, 1))


