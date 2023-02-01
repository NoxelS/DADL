import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable tensorflow warnings
from lib.cnn import CNN
from lib.utils import gram_cam

# Create the model
classifier = CNN(
    display_name="CatDogCNN",
    keep_checkpoints=True,
    load_path="models/catdog_cnn_model.h5",
    image_size=(200, 200),
    layer_array=[
        tf.keras.layers.Input((200, 200, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(16, (16, 16), activation='relu'),
        tf.keras.layers.MaxPool2D((4, 4), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (32, 32), activation='relu'),
        tf.keras.layers.MaxPool2D((4, 4), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation='sigmoid')
    ]
)

#classifier.test_dir(r"data/catdog/dataset/test_set/")

gram_cam(classifier, r"data/catdog/dataset/test_set/cats/cat.4001.jpg", "conv2d_1")

# classifier.fit_image_dir(
#     r"data/catdog/dataset/training_set/", 
#     r"data/catdog/dataset/test_set/", 
#     epochs=2, 
#     batch_size=64, 
#     verbose=1
# )

