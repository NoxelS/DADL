import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(trainx , trainy), (testx, testy) = tf.keras.datasets.mnist.load_data()

trainx = trainx / 255.0
testx = testx / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # 784
    tf.keras.layers.Dense(128, activation="relu"),  # 128
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")  # 10
])

model.compile(
    optimizer="SGD",
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)

histob = model.fit(trainx, trainy, epochs=10)

prediction = model.predict(tf.expand_dims(testx[5], 0))

print(model.evaluate(testx, testy))

print(prediction, testy[5])