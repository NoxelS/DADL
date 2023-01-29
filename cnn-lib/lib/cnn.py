from callbacks import UpdateTrainingBoard, PlotTrainingHistory
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable tensorflow warnings


class CNN(tf.keras.Model):
    def __init__(self, display_name, keep_checkpoints=False, x_train=None, y_train=None, x_test=None, y_test=None):
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
            tf.keras.layers.Conv2D(128, (4, 4), activation='relu', input_shape=(28, 28, 1)), # Batchsize x 28 x 28 x 1
            tf.keras.layers.MaxPool2D((2, 2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, 4, activation='relu'), 
            tf.keras.layers.MaxPool2D((2, 2), padding='same'),
            tf.keras.layers.BatchNormalization(), 
            tf.keras.layers.Conv2D(32, 4, activation='relu'), 
            tf.keras.layers.MaxPool2D((2, 2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
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


    def fit(self, ep=150, bs=64, verbose=1):

        # Data augmentation
        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.05,
            zoom_range=0.1,
            fill_mode='nearest'
        )

        # Make sure to backup after each epoch
        self.backup_callback = tf.keras.callbacks.BackupAndRestore(
            backup_dir=self.checkpoint_path + self.display_name,
            save_freq='epoch',
            delete_checkpoint=not self.keep_checkpoints,
            save_before_preemption=False
        )

        self.hist = self.model.fit(
            self.datagen.flow(
                self.x_train,
                self.y_train,
                batch_size=bs,
                shuffle=True
            ),
            validation_data=(self.x_test, self.y_test),
            epochs=ep,
            verbose=verbose,
            callbacks=[self.backup_callback, PlotTrainingHistory()]
        )

    def test(self):
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f'CNN-Test: acc = {100*acc:5.2f}%')


if __name__ == '__main__':
    # Load and prepare the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension because the model expects a 4D input (batchsize x height x width x channels)
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # Create a CNN instance
    cnn = CNN('mnist', keep_checkpoints=True, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    
    # Train the model
    cnn.fit()

