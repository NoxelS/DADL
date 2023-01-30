from callbacks import UpdateTrainingBoard, PlotTrainingHistory
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable tensorflow warnings


class CNN(tf.keras.Model):
    def __init__(self, display_name, keep_checkpoints=False, x_train=None, y_train=None, x_test=None, y_test=None, load_path=None):
        super(CNN, self).__init__()
        self.display_name = display_name
        self.keep_checkpoints = keep_checkpoints

        # Data
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        # Define model architecture
        self.layer_array = [
            tf.keras.layers.Conv2D(16, (4, 4), activation='relu', input_shape=(28, 28, 1)), # Batchsize x 28 x 28 x 1
            tf.keras.layers.MaxPool2D((2, 2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (4, 4), activation='relu'), # Batchsize x 28 x 28 x 1
            tf.keras.layers.MaxPool2D((2, 2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ]

        # Print model summary
        self.model = tf.keras.Sequential(self.layer_array)
        self.model.summary()

        # Compile the model
        self.model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        # Load weights if path is given
        if load_path is not None:
            self.model.load_weights(load_path)

        # Checkpoints to save trained network weights
        self.checkpoint_path = r"backup/"


    def init_weights(self):
        """
            Initialize weights with random values from a normal distribution
            between 0 and 1
        """
        weights_list = self.models.get_weights()
        for i in range(self.layer_array):
            weights_list[i] = np.random.normal(0, 1, size=weights_list[i].shape)
        self.models.set_weights(weights_list)


    def fit(self, epochs=150, batch_size=64, verbose=1, early_stop=False):

        # Data augmentation
        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.05,
            zoom_range=0.1,
            fill_mode='nearest'
        )

        # Make sure to backup after each epochs
        self.backup_callback = tf.keras.callbacks.BackupAndRestore(
            backup_dir=self.checkpoint_path + self.display_name,
            save_freq='epoch',
            delete_checkpoint=not self.keep_checkpoints,
            save_before_preemption=False
        )

        # Reduce learning rate if no improvement after 10 epochs
        self.reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=10,
            verbose=0,
            mode='auto',
            min_delta=0.0001,
            cooldown=0,
            min_lr=0.0001
        )

        # Early stopping if no improvement after 20 epochs and restore best weights
        self.early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=20,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=True
        )

        callbacks = [self.backup_callback, PlotTrainingHistory(), self.reduce_lr]

        if early_stop:
            callbacks.append(self.early_stopping)

        self.hist = self.model.fit(
            self.datagen.flow(
                self.x_train,
                self.y_train,
                batch_size=batch_size,
                shuffle=True
            ),
            validation_data=(self.x_test, self.y_test),
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks
        )


    def test(self):
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f'CNN-Test: acc = {100*acc:5.2f}%')


    def weights_saturation(self):
        """
            Returns the mean and standard deviation of the weights
        """
        weights_list = self.model.get_weights()
        mean = 0
        std = 0
        for i in range(len(weights_list)):
            mean += np.mean(weights_list[i])
            std += np.std(weights_list[i])
        return mean / len(weights_list), std / len(weights_list)
    
    def activation_map(self, x, layer_name):
        """
            Returns the activation map of a given input for a given layer
        """
        for layer in self.model.layers:
            if layer.name == layer_name:
                return layer.call(x)
            else:
                x = layer.call(x)

if __name__ == '__main__':
    # Load and epochs the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension because the model expects a 4D input (batchsize x height x width x channels)
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # Create a CNN instance
    cnn = CNN('mnist', keep_checkpoints=True, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    # Load model
    cnn.model.load_weights('mnist.h5')
    # Train the model
    # cnn.fit()


    # cnn.fit(epochs=2)
    cnn.test()

    # cnn.model.save('mnist.h5')
    activation_imgaes = []
    for i in range(10):
        # Find a test image that is calssidied as i
        for j in range(len(y_test)):
            if y_test[j] == i:
                print(f'Class {i}: {j} / {len(y_test)}')
                # Reshape the image to a 4D tensor
                x_tmp = x_test[j]
                x_tmp = x_tmp.reshape(1, 28, 28, 1)
                x_tmp = x_tmp.astype('float32') # Conv2d expects float32
                utils.plot_activations(x_tmp, cnn, ['conv2d', 'conv2d_1', 'dense_1'], path=f'figures/activations_{i}.png')
                break