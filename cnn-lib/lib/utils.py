import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tensorflow import keras
import os
from lib.cnn import *

# Display
from IPython.display import Image, display
import matplotlib.cm as cm

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


def get_img_array(img_path, size):
    img = keras.preprocessing.image.load_img(img_path, target_size=size, color_mode='grayscale')
    # `array` is a float32 Numpy array of shape (200, 200, 1)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 200, 200, 1)
    array = np.expand_dims(array, axis=0)
    return array



def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def gram_cam(cnn, img_path, layer_name):
    # Image size is possible != to cnn.image_size so we need to resize it an save it
    img = keras.preprocessing.image.load_img(img_path, target_size=cnn.image_size)
    img.save(img_path)

    model_builder = keras.applications.xception.Xception
    preprocess_input = keras.applications.xception.preprocess_input
    decode_predictions = keras.applications.xception.decode_predictions

    # Prepare image
    img_array = preprocess_input(get_img_array(img_path, size=cnn.image_size))

    # Make model
    model = cnn.model

    # Remove last layer's softmax
    model.layers[-1].activation = None

    # Print what the top predicted class is
    print(img_array.shape)
    preds = model.predict(img_array)
    print("Predicted:", decode_predictions(preds, top=1)[0])

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, model, layer_name)

    # Display heatmap
    plt.matshow(heatmap)
    plt.show()
