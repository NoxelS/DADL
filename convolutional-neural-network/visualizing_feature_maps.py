import tensorflow as tf
from keras import datasets
from keras.models import Model

import matplotlib.pyplot as plt

import numpy as np

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images, test_images = train_images / 255.0 , test_images / 255.0

cnn = tf.keras.models.load_model("models/aug_cnn_model.h5")

#Für die Nummerierung der layer Benutzen wir die Nummerierung, die Christian in dem Skript bei seiner
#Einführung in die Welt der neuronalen Netzwerke angewandt hat.
def shortening_model_to_layer(layer, model):
    hidden_layers = [model.layers[i].output for i in range(0,layer)]
    shortened_model = Model(inputs = model.inputs, outputs = hidden_layers)
    return shortened_model

def activation_maps_up_to_last_layer(shortened_model, image):
    return shortened_model.predict(image)

def show_activation_maps_of_desired_layer(all_activation_maps, layer):
    if layer != 1:
        desired_activation_map = all_activation_maps[layer-1]
    else:
        desired_activation_map = all_activation_maps

    for i in range(desired_activation_map.shape[-1]):
        ax = plt.subplot(1, desired_activation_map.shape[-1], i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(desired_activation_map[0,:,:,i]) 
    plt.show()

shortened_model = shortening_model_to_layer(3, cnn)
activation_maps = activation_maps_up_to_last_layer(shortened_model, np.array([train_images[6]]))
show_activation_maps_of_desired_layer(activation_maps, 3)

