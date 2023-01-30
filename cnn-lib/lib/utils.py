import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import cnn


def plot_activations(img, cnn, layer_names, path=None):
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