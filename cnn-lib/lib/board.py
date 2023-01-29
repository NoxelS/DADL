import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image
import numpy as np
from scipy import ndimage

class Board(tk.Frame):
    def __init__(self, parent, title, geometry, message):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.parent.title(title)
        self.parent.geometry(geometry)

        self.previous_x = self.previous_y = 0
        self.x = self.y = 0
        self.points_recorded = []

        tk.Label(self, text=message).grid()
        self.canvas = tk.Canvas(self, width=280, height=280, bg = "black", cursor="cross")
        self.canvas.grid(column=0, row=0)
        self.canvas.bind("<Motion>", self.update_mouse_position)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.update_plots)

        # Add row to add buttons to
        self.action_row = tk.Frame(self, width=280, height=50)
        self.action_row.grid(column=0, row=2, columnspan=2)
        # Add buttons to the action row
        self.clear_button = tk.Button(self.action_row, text="Clear", command=self.clear_all)
        self.clear_button.grid(column=0, row=0)
        # self.print_button = tk.Button(self.action_row, text="Print", command=self.print_points)
        # self.print_button.grid(column=1, row=0)


        # Initialize the image array
        self.image = np.zeros((280, 280), dtype=np.uint8)

        # Add plot areas
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        self.fig1 = Figure(figsize = (280*px, 280*px),
                 dpi = 100, constrained_layout=True)
        self.fig2 = Figure(figsize = (280*px, 280*px),
                 dpi = 100, constrained_layout=True)
        self.fig3 = Figure(figsize = (280*px, 280*px),
                 dpi = 100, constrained_layout=True)
  
        y = [i**2 for i in range(101)]

        # adding the subplots
        plot1 = self.fig1.add_subplot(111)
        plot1.plot(y)
        plot2 = self.fig2.add_subplot(111)
        plot2.plot(y)
        plot3 = self.fig3.add_subplot(111)
        plot3.plot(y)

        # containing the Matplotlib figure
        self.pt1 = FigureCanvasTkAgg(self.fig1,
                                master = self)  
        self.pt2 = FigureCanvasTkAgg(self.fig2,
                                master = self)
        self.pt3 = FigureCanvasTkAgg(self.fig3,
                                master = self)
        self.pt1.draw()
        self.pt2.draw()
        self.pt3.draw()

    
        # placing the canvas on the Tkinter window
        self.pt1.get_tk_widget().grid(row=0, column=1)
        self.pt2.get_tk_widget().grid(row=1, column=0)
        self.pt3.get_tk_widget().grid(row=1, column=1)


        self.pack(side="top", fill="both", expand=True)

    def update_plots(self, event=None):
        # Downscale to 28x28
        self.image2 = ndimage.zoom(self.image, 0.1, order=0)
        # Plot the image array in the first plot
        self.fig1.axes[0].clear()
        self.fig1.axes[0].imshow(ndimage.gaussian_filter(self.image2, sigma=1), cmap='gray')
        # Hide the axes
        self.fig1.axes[0].axis('off')
        # Add title
        self.fig1.axes[0].set_title("Gaussian Image")

        self.pt1.draw()

    def clear_all(self, event=None):
        self.canvas.delete("all")
        self.image = np.zeros((280, 280), dtype=np.uint8)
        self.update_plots()


    def update_mouse_position(self, event):
        self.previous_x = event.x
        self.previous_y = event.y

    def draw(self, event):
        radius = 10

        if self.points_recorded:
            self.points_recorded.pop()
            self.points_recorded.pop()

        self.x = event.x
        self.y = event.y
        self.points_recorded.append(self.previous_x)
        self.points_recorded.append(self.previous_y)
        self.points_recorded.append(self.x)
        self.points_recorded.append(self.x)
        self.previous_x = self.x
        self.previous_y = self.y

        # Add the drawn line to the image array as a white line
        for x in range(-radius, radius, 1):
            for y in range(-radius, radius, 1):
                if x**2 + y**2 <= radius**2 and self.x + x >= 0 and self.x + x < 280 and self.y + y >= 0 and self.y + y < 280:
                    self.image[self.previous_y + y, self.previous_x + x] = 255
                    self.image[self.y + y, self.x + x] = 255

        self.canvas.create_line(self.previous_x, self.previous_y,
                                self.x, self.y, fill="yellow", width=radius*2, capstyle="round", smooth=False)


def main():
    root = tk.Tk()
    Board(root, "Board", "560x840", "Hello World!")
    root.mainloop()

if __name__ == "__main__":
    main()