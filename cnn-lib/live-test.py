from lib.cnn import CNN
from lib.board import Board
import tkinter as tk
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable tensorflow warnings


# Create a CNN instance
cnn = CNN('mnist', load_path='models/mnist.h5')

# Create board
root = tk.Tk()
Board(root, "Board", "560x840", "MNIST Live Test", cnn)
root.mainloop()