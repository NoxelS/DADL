import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from keras.layers import Input, Dense, Lambda, Reshape, Flatten
from keras.models import Model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import keras.backend as K

k=2     # latent space dimension
c=1.    # parameter for the loss function
batch_size=50

# Load the data
(xtrain, _), (xtest, ytest) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize the data
xtrain = xtrain / 255.
xtest = xtest / 255.


def sample(args):
    z_mean, z_log_var = args    # Shape: (k, )
    epsilon = K.random_normal(shape=(k, ), mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon


# Create encoder
x = Input(shape=(28, 28))
xflat = Flatten()(x)
enc_hidden = Dense(512, activation='relu')(xflat)
z_mean = Dense(k, activation='linear')(enc_hidden)
z_log_var = Dense(k, activation='linear')(enc_hidden)
encoder = Model(x, [z_mean, z_log_var])

# Weights for VAE
dec_hidden_weights = Dense(512, activation='relu')
x_out_weights = Dense(784, activation='linear')

# Create decoder
z = Input(shape=(k, ))
dec_hidden = dec_hidden_weights(z)
x_out_flat = x_out_weights(dec_hidden)
x_out = Reshape((28, 28))(x_out_flat)
decoder = Model(z, x_out)

# Create VAE
_z = Lambda(sample)([z_mean, z_log_var])
_dec_hidden = dec_hidden_weights(_z)
_x_out_flat = x_out_weights(_dec_hidden)
_x_out = Reshape((28, 28))(_x_out_flat)
vae = Model(x, _x_out)

# Loss function
def vae_loss(x, _x_out):        # shape (batch_size, 28, 28)
    mean, log_var = encoder(x)  # shape (batch_size, k)
    x = K.flatten(x)        # shape (batch_size, 784)
    _x_out = K.flatten(_x_out)  # shape (batch_size, 784)

    # Reconstruction loss
    rec_loss = K.sum(K.square(x - _x_out), axis=-1)/(2*c)

    # Kl divergence
    kl_loss = 0.5 * K.sum(K.exp(log_var) + K.square(mean) - 1. - mean, axis=-1)

    return rec_loss + kl_loss


vae.compile(optimizer='adam', loss=vae_loss)
vae.fit(xtrain, xtrain, epochs=10, batch_size=batch_size)

# Plot the results
z,_ = encoder.predict(xtest)
plt.figure(figsize=(12,10))
plt.scatter(z[:,0], z[:,1], c=ytest, cmap='tab10')
plt.colorbar()
plt.xlabel("z[0]")
plt.ylabel("z[1]")


# Generate new images from random samples

def plot_latent_images(m):
    norm = tfp.distributions.Normal(0, 1)
    grid_x = norm.quantile(np.linspace(0.05, 0.95, m))
    grid_y = norm.quantile(np.linspace(0.05, 0.95, m))

    image = np.zeros((28*m, 28*m))
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(28, 28)
            image[i * 28: (i + 1) * 28,
                  j * 28: (j + 1) * 28] = digit
            
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')

plot_latent_images(10)

plt.show()
