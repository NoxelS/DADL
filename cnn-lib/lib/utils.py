import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os



def plot_activations(img, cnn, layer_names, path=None):
    """
        Plots the activation maps of a given input for a given layer
    """

    activations = [cnn.activation_map(img, ln) for ln in layer_names]
    cols = len(layer_names) + 1
    rows = max([a.shape[3] for a in activations if len(a.shape) != 2])

    # Make grid figure
    gs = gridspec.GridSpec(rows, cols, wspace=0.0, hspace=0.0, top=1.0, bottom=0.0, left=0.0, right=1.0)

    # Plot input image along first column
    for i in range(rows):
        ax = plt.subplot(gs[i, 0])
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
        else:
            # Convolutional layer
            for j in range(activations[i].shape[3]):
                ax = plt.subplot(gs[j, i + 1])
                ax.imshow(activations[i][0, :, :, j])
                ax.set_aspect('equal')
                ax.axis('off')

    if path:
        plt.savefig(path)
    else:
        plt.show()