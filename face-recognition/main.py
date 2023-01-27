import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import time

(x_train, y_train) = (None, None)

# Read all the images from the folder


def read_images_grey(path, n=None):
    return [cv2.resize(cv2.imread(path + file, cv2.IMREAD_GRAYSCALE), (100, 100)) for file in os.listdir(path)[0: n or len(os.listdir(path))]]


woman_n = 9000 # len(os.listdir("faces/woman/"))
man_n = 9000 # len(os.listdir("faces/man/"))

womam_images = read_images_grey("faces/woman/", woman_n)
man_images = read_images_grey("faces/man/", man_n)

x_train = tf.convert_to_tensor(womam_images + man_images)
print(x_train.shape)
x_train = tf.cast(x_train, tf.float32)

y_train = tf.convert_to_tensor(
    [[1., 0.] for _ in range(woman_n)] + [[0., 1.] for _ in range(man_n)])

# Normalize the images
x_train = x_train / 255.0

# Shuffle the data
perms = np.random.permutation(len(x_train))
x_train = tf.convert_to_tensor([x_train[i] for i in perms])
y_train = tf.convert_to_tensor([y_train[i] for i in perms])

# Use the first 20% of the data for validation
val_x = x_train[0: int(len(x_train) * 0.2)]
val_y = y_train[0: int(len(y_train) * 0.2)]

# Use the rest of the data for training
x_train = x_train[int(len(x_train) * 0.2):]
y_train = y_train[int(len(y_train) * 0.2):]

# Create the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (2, 2), activation='relu', input_shape=(100, 100, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (2, 2), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(8, (2, 2), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
# model.fit(x_train, y_train, epochs=10)

# Save the model
# model.save("face-recognition-model")



# Load the model
model = tf.keras.models.load_model("face-recognition-model")

# Save model a h5
model.save("face-recognition-model.h5")

# Test the model with a real image
test_x = read_images_grey("faces/test/", 1)
print(test_x[0].shape)

test_x = tf.convert_to_tensor(test_x)
test_x = tf.cast(test_x, tf.float32)
test_x = test_x / 255.0

# Evaluate the model
print(model.evaluate(val_x, val_y))

# Predict the result
print(model.predict(test_x))

