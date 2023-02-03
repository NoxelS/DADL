import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tensorflow import keras
from lib.cnn import *
import keras
import scipy

# Display
from IPython.display import Image, display
import matplotlib.cm as cm

def plot_activations(img, cnn, layer_names, path=None, hide_original=False):
    """
        Plots the activation maps of a given input for a given layer
    """
    has_dense = False

    activations = [cnn.activation(img, ln) for ln in layer_names]
    cols = len(layer_names) + 1
    rows = max([a.shape[3] for a in activations if len(a.shape) != 2])

    # Make grid figure
    gs = gridspec.GridSpec(rows, cols, wspace=0.0, hspace=0.0, top=1.0, bottom=0.0, left=0.0, right=1.0)


    # Plot input image in the center
    if not hide_original:
        ax = plt.subplot(gs[(rows - 1) // 2, 0])
        ax.imshow(img[0, :, :, 0])
        ax.set_aspect('equal')
        ax.axis('off')


    # Plot activation maps
    for i, layer_name in enumerate(layer_names):
        # Check if layer is dense or convolutional
        if len(activations[i].shape) == 2:
            # Dense layer
            ax = plt.subplot(gs[:, i + 1])
            ax.barh(np.arange(activations[i].shape[1]), activations[i][0, :])
            ax.set_yticks(np.arange(activations[i].shape[1]))
            has_dense = True
        else:
            # Convolutional layer
            start = (rows - activations[i].shape[3]) // 2
            for j in range(start, start + activations[i].shape[3]):
                ax = plt.subplot(gs[j, i + 1])
                ax.imshow(activations[i][0, :, :, j - start])
                ax.set_aspect('equal')
                ax.axis('off')

    # Set figure size
    fig = plt.gcf()
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches

    
    width = cols * 2 * 28 * px
    height = rows * 2 * 28 * px
    if has_dense:
        width = width + 250 * px

    fig.set_size_inches(width , height)


    if path:
        plt.savefig(path)
    else:
        plt.margins(0, 0)
        plt.show()


def plot_missclassified_images(x_test, y_test, cnn, path):
    """
        Plots the missclassified images of a given CNN
    """
    # Get missclassified images
    y_pred = cnn.model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    missclassified = np.where(y_pred != y_test)[0]
    print(len(missclassified), "missclassified images")
    # Plot missclassified images
    fig = plt.figure(figsize=(10, 10))
    for i, idx in enumerate(missclassified):
        ax = fig.add_subplot(10, 10, i + 1)
        ax.imshow(x_test[idx, :, :, 0], cmap='gray')
        # invert colors
        ax.imshow(1 - ax.images[0].get_array(), cmap='gray')
        
        ax.set_title("p:" + str(y_pred[idx]) + "/t:" + str(y_test[idx]))
        ax.axis('off')
    fig.tight_layout()
    plt.savefig(path)


def get_class_activation_map(model, img):
    img = np.expand_dims(img, axis=0) # 1 x 28 x 28 x 1

    # predict to get the winning digit
    predictions = model.predict(img)
    label_index = np.argmax(predictions)

    # Get input weights to the softmax of the winning class.
    class_weights = model.layers[-1].get_weights()[0]
    class_weights_winner = class_weights[:, label_index]
    
    # get the final conv layer
    final_conv_layer = model.get_layer("conv2d_1")
    
    # create a function to fetch the final conv layer output maps
    get_output = keras.backend.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img])
    
    # squeeze conv map to shape image to size 
    conv_outputs = np.squeeze(conv_outputs) # 24 x 24 x 32
    
    # bilinear upsampling to resize each filtered image to size of original image 
    mat_for_mult = scipy.ndimage.zoom(
        conv_outputs, (28/24, 28/24, 1), order=1)  # dim: 28 x 28 x 32

    # get class activation map for object class that is predicted to be in the image
    final_output = np.dot(mat_for_mult.reshape((28*28, 32)), class_weights_winner).reshape(28,28)

    return final_output


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

# shortened_model = shortening_model_to_layer(3, cnn)
# activation_maps = activation_maps_up_to_last_layer(shortened_model, np.array([train_images[6]]))
# show_activation_maps_of_desired_layer(activation_maps, 3)


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

# kernel_visualisieren(kernel(cnn, 1))
