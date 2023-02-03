import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

class PlotTrainingHistory(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):

        loss_y = []
        val_loss_y = []
        acc_y = []
        val_acc_y = []

        # Training history
        if not os.path.exists(r'training'):
            os.makedirs(r'training')

        # Clear the files if it's the first epoch
        if epoch == 0:
            for file in os.listdir(r'training'):
                if file.endswith('.png') or file.endswith('.dat'):
                    os.remove(os.path.join(r'training', file))

        # Save current epoch data and fetch history
        with open(r'training/acc.dat', 'a') as f:
            f.write(f'\n{epoch},{logs["accuracy"]},{logs["val_accuracy"]}')
        
        with open(r'training/acc.dat', 'r') as f:
            for line in f:
                if len(line.split(',')) < 3:
                    continue
                acc_y.append(float(line.split(',')[1]))
                val_acc_y.append(float(line.split(',')[2]))

        
        with open(r'training/loss.dat', 'a') as f:
            f.write(f'\n{epoch},{logs["loss"]},{logs["val_loss"]}')
        
        with open(r'training/loss.dat', 'r') as f:
            for line in f:
                if len(line.split(',')) != 3:
                    continue
                loss_y.append(float(line.split(',')[1]))
                val_loss_y.append(float(line.split(',')[2]))


        # Plot loss
        plt.plot(loss_y)
        plt.plot(val_loss_y)
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'])
        plt.savefig(r'training/loss.png')
        plt.clf()

        # Plot accuracy
        plt.plot(acc_y)
        plt.plot(val_acc_y)
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(r'training/accuracy.png')
        plt.clf()