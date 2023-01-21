import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(xtrain, _), (xtest, _) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize the data
xtrain = xtrain / 255.
xtest = xtest / 255.

# Add noise
xtrain_noise = (xtrain + 0.2 * tf.random.normal(shape=xtrain.shape))/1.2
xtest_noise = (xtest + 0.2 * tf.random.normal(shape=xtest.shape))/1.2

layer_size = [784, 144, 10, 144, 784]

# Create autoencoder
class Autoencoder(tf.keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(layer_size[1], activation='relu'),
            tf.keras.layers.Dense(layer_size[2], activation='relu')
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(layer_size[3], activation='relu'),
            tf.keras.layers.Dense(layer_size[4], activation='sigmoid'),
            tf.keras.layers.Reshape((28, 28))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

dae1 = Autoencoder()

dae1.compile(optimizer='adam', loss='mse')
dae1.fit(xtrain_noise, xtrain, epochs=3, batch_size=256, shuffle=True, validation_data=(xtest_noise, xtest))

encoded_imgs= dae1.encoder(xtest_noise).numpy()
decoded_imgs= dae1.decoder(encoded_imgs)

# Plot the images
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(xtest_noise[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()