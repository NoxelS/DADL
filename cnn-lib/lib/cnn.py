from lib.callbacks import *
from lib.utils import *
import numpy as np
import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable tensorflow warnings


class CNN(tf.keras.Model):
    def __init__(
            self,
            display_name, 
            keep_checkpoints=False, 
            x_train=None,
            y_train=None,
            x_test=None,
            y_test=None,
            load_path=None, 
            layer_array=None,
            image_size=(28, 28)
        ):
        super(CNN, self).__init__()
        self.display_name = display_name
        self.keep_checkpoints = keep_checkpoints
        self.image_size = image_size
        self.load_path = load_path

        # Data
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        # Define model architecture
        if layer_array is not None:
            self.layer_array = layer_array
        else:
            # Standard CNN architecture (MNIST)
            self.layer_array = [
                tf.keras.layers.Input((image_size[0], image_size[1], 1)), # Keras will automatically add a dimension for the batch size
                tf.keras.layers.Conv2D(16, (4, 4), activation='relu'),
                tf.keras.layers.MaxPool2D((2, 2), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(32, (4, 4), activation='relu'),
                tf.keras.layers.MaxPool2D((2, 2), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation='softmax')
            ]

        # Print model summary
        self.model = tf.keras.Sequential(self.layer_array, name=self.display_name)
        self.model.summary()

        # Save model architecture if pydot and graphviz are installed
        try:
            dot_img_file = r'figures/model.png'
            plot_model(self.model, to_file=dot_img_file, show_shapes=True)
        except Exception as e:
            print("Missing pydot or graphviz. Unable to save model architecture as image.")

        # Compile the model
        self.model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        # Load weights if path is given
        if load_path is not None and os.path.exists(load_path):
            self.model.load_weights(load_path)
            print("Loaded weights from " + load_path)

        # Checkpoints to save trained network weights
        self.checkpoint_path = r"backup/"


    def init_weights(self):
        """
            Initialize weights with random values from a normal distribution
            between 0 and 1
        """
        print("INIT WEIGHTS HERE")
        weights_list = self.models.get_weights()
        for i in range(self.layer_array):
            weights_list[i] = np.random.normal(0, 1, size=weights_list[i].shape)
        self.models.set_weights(weights_list)


    def fit(self, epochs=150, batch_size=64, verbose=1, early_stop=False):

        # Data augmentation
        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=25,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.05,
            zoom_range=0.1,
        )

        # Make sure to backup after each epochs
        self.backup_callback = tf.keras.callbacks.BackupAndRestore(
            backup_dir=self.checkpoint_path + self.display_name,
            save_freq='epoch',
            delete_checkpoint=not self.keep_checkpoints,
            save_before_preemption=False
        )

        # Reduce learning rate if no improvement after 8 epochs
        self.reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.75,
            patience=8,
            verbose=0,
            mode='max',
            min_delta=0.0005,
            cooldown=0,
            min_lr=0.000001
        )

        # CSV logger to save training history
        self.csv_logger = tf.keras.callbacks.CSVLogger(
            f"training_history_{self.display_name}.csv",
            separator=',',
            append=True
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


    def fit_image_dir(self, train_dir, test_dir, epochs=150, batch_size=64, verbose=1, early_stop=False, target_shape=None):
        """
            Used to fit the model with images from a directory by using generator to prevent memory overflow.
        """
        if target_shape is None:
            target_shape = self.image_size

        # Data Augmentation for training data
        train_images_generator = ImageDataGenerator(rotation_range=25,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.05,
            zoom_range=0.1,
            rescale=1.0/255
        )

        # Training data generator
        train_data_gen = train_images_generator.flow_from_directory(
            batch_size=batch_size,
            directory=train_dir,
            shuffle=True,
            target_size=target_shape,
            class_mode='categorical',
            color_mode='grayscale'
        )

        # Test data generator (no data augmentation)
        test_images_generator = ImageDataGenerator(rescale=1.0/255)
        test_data_gen = test_images_generator.flow_from_directory(
            batch_size=batch_size,
            directory=test_dir,
            shuffle=True,
            target_size=target_shape,
            class_mode='categorical',
            color_mode='grayscale'
        )

        # Make sure to backup after each epoch
        self.backup_callback = tf.keras.callbacks.BackupAndRestore(
            backup_dir=self.checkpoint_path + self.display_name,
            save_freq='epoch',
            delete_checkpoint=not self.keep_checkpoints,
            save_before_preemption=False
        )

        # Reduce learning rate if no improvement after 8 epochs
        self.reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.75,
            patience=8,
            verbose=0,
            mode='max',
            min_delta=0.0005,
            cooldown=0,
            min_lr=0.000001
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

        # Fit with data flow from directory
        self.hist = self.model.fit_generator(
            train_data_gen,
            validation_data=test_data_gen,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks
        )


    def test(self):
        """
            Test the model with test data
        """
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f'CNN-Test: acc = {100*acc:5.2f}%')


    def test_dir(self, dir):
        """
            Test the model with images from a directory by using generator to prevent memory overflow
        """
        
        test_images_generator = ImageDataGenerator(rescale=1.0/255)
        test_data_gen = test_images_generator.flow_from_directory(
            batch_size=1,
            directory=dir,
            shuffle=True,
            target_size=self.image_size,
            class_mode='categorical',
            color_mode='grayscale'
        )
        loss, acc = self.model.evaluate_generator(test_data_gen, verbose=0)
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


    def activation(self, x, layer_name):
        """
            Returns the activation map of a given input for a given layer
        """
        for layer in self.model.layers:
            if layer.name == layer_name:
                return layer.call(x)
            else:
                x = layer.call(x)

    def save(self, path):
        """
            Saves the model
        """
        self.model.save(path)