import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the data
(xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.mnist.load_data()

# Normalize the data
xtrain, xtest = xtrain / 255., xtest / 255.

# Show the first 10 images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(xtrain[i], cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(ytrain[i])
plt.tight_layout()
plt.savefig('./figures/10-mnist.png')
# Create a sequential model without cnn
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Display the model summary
model.summary()

# Check if file exits
if os.path.exists('./models/nn_model.h5'):
    model = tf.keras.models.load_model('./models/nn_model.h5')
else:
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(xtrain, ytrain, epochs=10, batch_size=256, validation_data=(xtest, ytest))

    # Save the model
    model.save('./models/nn_model.h5')

# Get images that are misclassified
y_pred = model.predict(xtest)
misclassified = np.where(y_pred.argmax(axis=1) != ytest)[0]

print(f"NN: Number of misclassified images: {len(misclassified)}")

# Show the first 25 misclassified images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(xtest[misclassified[i]], cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Actual: ' + str(ytest[misclassified[i]]) + ' Predicted: ' + str(y_pred.argmax(axis=1)[misclassified[i]]))

plt.tight_layout()
plt.savefig('./figures/NN-10-mnist-misclassified.png')

# Create a sequential model with cnn
cnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (2, 2), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (2, 2), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Display the model summary
cnn_model.summary()

# Check if file exits
if os.path.exists('./models/cnn_model.h5'):
    cnn_model = tf.keras.models.load_model('./models/cnn_model.h5')
else:
    # Compile the model
    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(xtrain.reshape(-1, 28, 28, 1), ytrain, epochs=10, batch_size=256, validation_data=(xtest.reshape(-1, 28, 28, 1), ytest))

    # Save the model
    cnn_model.save('./models/cnn_model.h5')

# Get images that are misclassified
y_pred = cnn_model.predict(xtest.reshape(-1, 28, 28, 1))
misclassified = np.where(y_pred.argmax(axis=1) != ytest)[0]

print(f"CNN: Number of misclassified images: {len(misclassified)}")

# Show all the misclassified images
plt.figure(figsize=(10,10))
for i in range(len(misclassified)):
    plt.subplot(int(np.ceil(np.sqrt(len(misclassified)))),int(np.ceil(np.sqrt(len(misclassified)))), i+1)
    plt.imshow(xtest[misclassified[i]], cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(f"{ytest[misclassified[i]]}/{y_pred.argmax(axis=1)[misclassified[i]]}")

plt.tight_layout()
plt.savefig('./figures/CNN-mnist-misclassified.png')

# Create a sequential model with cnn and data augmentation

# Create a data generator
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode='nearest'
)

# Create a sequential model with cnn
aug_cnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (2, 2), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (2, 2), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Check if file exits
if os.path.exists('./models/aug_cnn_model.h5'):
    aug_cnn_model = tf.keras.models.load_model('./models/aug_cnn_model.h5')
else:
    # Create a callback to save the model after each epoch
    callback = tf.keras.callbacks.BackupAndRestore(
        './backups',
        save_freq='epoch',
        delete_checkpoint=True,
        save_before_preemption=False
    )

    # Compile the model
    aug_cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    aug_cnn_model.fit(
        datagen.flow(
            xtrain.reshape(-1, 28, 28, 1), 
            ytrain, 
            batch_size=256
        ), 
        epochs=250, 
        validation_data=(xtest.reshape(-1, 28, 28, 1), ytest),
        callbacks=[callback]
    )

    # Save the model
    aug_cnn_model.save('./models/aug_cnn_model.h5')

    # Plot the training history
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(aug_cnn_model.history.history['loss'], label='loss')
    # plt.plot(aug_cnn_model.history.history['val_loss'], label='val_loss')
    # plt.legend()
    # plt.subplot(1, 2, 2)
    # plt.plot(aug_cnn_model.history.history['accuracy'], label='accuracy')
    # plt.plot(aug_cnn_model.history.history['val_accuracy'], label='val_accuracy')
    # plt.ylim(0.99, 1.0)
    # plt.legend()
    # plt.savefig('./figures/CNN-mnist-training-history.png')

# Get images that are misclassified
y_pred = aug_cnn_model.predict(xtest.reshape(-1, 28, 28, 1))
misclassified = np.where(y_pred.argmax(axis=1) != ytest)[0]

print(f"AUG CNN: Number of misclassified images: {len(misclassified)}")

# Show all the misclassified images
plt.figure(figsize=(10,10))
for i in range(len(misclassified)):
    plt.subplot(int(np.ceil(np.sqrt(len(misclassified)))),int(np.ceil(np.sqrt(len(misclassified)))), i+1)
    plt.imshow(xtest[misclassified[i]], cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(f"{ytest[misclassified[i]]}/{y_pred.argmax(axis=1)[misclassified[i]]}")

plt.tight_layout()
plt.savefig('./figures/AUG-CNN-mnist-misclassified.png')