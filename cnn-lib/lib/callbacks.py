import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt



# Use this callback to update plots in the GUI
class UpdateTrainingBoard(keras.callbacks.Callback):
    def __init__(self, training_board=None):
        super().__init__()
        self.training_board = training_board

    def on_epoch_end(self, epoch, logs=None):
        # self.training_board.update_plots()
        print("Epoch: ", epoch)
        keys = list(logs.keys())
        print(keys)


class PlotTrainingHistory(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):

        loss_y = []
        val_loss_y = []
        acc_y = []
        val_acc_y = []

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

if __name__ == '__main__':
    # Test the callback
    plot_training_history = PlotTrainingHistory()
    plot_training_history.on_epoch_end(9, {'accuracy': 0.5, 'val_accuracy': 0.5, 'loss': 0.5, 'val_loss': 0.5})
    # plot_training_history.on_epoch_end(2, {'accuracy': 0.4, 'val_accuracy': 0.2, 'loss': 0.1, 'val_loss': 0.4})
    # plot_training_history.on_epoch_end(3, {'accuracy': 0.3, 'val_accuracy': 0.1, 'loss': 0.01, 'val_loss': 0.2})
