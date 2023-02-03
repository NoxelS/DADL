from lib.board import Board
import tkinter as tk
from lib.utils import get_class_activation_map
from lib.cnn import CNN
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable tensorflow warnings
from keras import layers


# Load and epochs the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension because the model expects a 4D input (batchsize x height x width x channels)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Create a CNN instance
cnn = CNN('mnist-cam', keep_checkpoints=True, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
          load_path='models/mnist-cam.h5', layer_array=[
            layers.InputLayer(input_shape=(28, 28, 1)),
            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.GlobalAveragePooling2D(),
            layers.Dense(10, activation='softmax')
])

# Fit and save model
cnn.fit(epochs=88, batch_size=256)
cnn.save('models/mnist-cam.h5')

# get_class_activation_map(cnn.model, x_test[0])

# cam_data = [] # Tuples of (image, cam)
# for i in range(10):
#     # Find a test image that is calssidied as i
#     for j in range(len(y_test)):
#         if y_test[j] == i:
#             # Reshape the image to a 4D tensor
#             x_tmp = x_test[j]
#             cam = get_class_activation_map(cnn.model, x_tmp)
#             cam_data.append((x_tmp, cam))
#             break

# # Plot the images all in a 10*3 grid
# fig, axs = plt.subplots(10, 3)
# for i in range(10):
#     img = cam_data[i][0][:, :, 0]
#     cam = cam_data[i][1]

#     axs[i, 0].imshow(img, cmap='gray')
#     axs[i, 1].imshow(cam, cmap='jet')
#     axs[i, 2].imshow(img, cmap='gray')
#     axs[i, 2].imshow(cam, cmap='jet', alpha=0.5)

#     axs[i, 0].axis('off')
#     axs[i, 1].axis('off')
#     axs[i, 2].axis('off')

# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.show()


# Create board
root = tk.Tk()
Board(root, "Board", "560x920", "CAM Live Test", cnn, True)
root.mainloop()