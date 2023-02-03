from lib.utils import plot_missclassified_images, plot_activations
from lib.cnn import CNN
from lib.board import Board
import tensorflow as tf
import numpy as np
import tkinter as tk
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable tensorflow warnings


# Create a CNN instance
cnn = CNN('mnist', load_path='models/mnist.h5')

# Create board
root = tk.Tk()
Board(root, "Board", "560x840", "MNIST Live Test", cnn)
root.mainloop()