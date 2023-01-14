import numpy as np
import matplotlib.pyplot as plt

def convolve(image, kernel):
    # Get dimensions of image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate the output shape
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    
    # Convolve
    output = np.zeros((output_height, output_width))
    for x in range(output_height):
        for y in range(output_width):
            output[x][y] = np.sum(image[x:x+kernel_height, y:y+kernel_width] * kernel)
    
    return output

# Define kernels
identity_kernel = np.array([[0, 0, 0],
                            [0, 1, 0],  
                            [0, 0, 0]])

blur_kernel = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]]) / 9.0

sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

edge_detection_kernel = np.array([[-1, -1, -1],
                                  [-1, 8, -1],
                                  [-1, -1, -1]])

# Load image
image = plt.imread('assets/mona.jpeg') / 255.0

# Convolve image with kernels
identity_output = convolve(image, identity_kernel)
blur_output = convolve(image, blur_kernel)
sharpen_output = convolve(image, sharpen_kernel)
edge_detection_output = convolve(image, edge_detection_kernel)

# Plot all kernels 2x2
fig, axes = plt.subplots(2, 2, figsize=(5, 5))
axes[0, 0].imshow(identity_kernel)
axes[0, 0].set_title('Identity')
axes[0, 1].imshow(blur_kernel)
axes[0, 1].set_title('Blur')
axes[1, 0].imshow(sharpen_kernel)
axes[1, 0].set_title('Sharpen')
axes[1, 1].imshow(edge_detection_kernel)
axes[1, 1].set_title('Edge Detection')
fig.suptitle('Kernels', fontsize=16)

# Add colorbar
plt.colorbar(axes[0, 0].imshow(identity_kernel), ax=axes[0, 0])
plt.colorbar(axes[0, 1].imshow(blur_kernel), ax=axes[0, 1])
plt.colorbar(axes[1, 0].imshow(sharpen_kernel), ax=axes[1, 0])
plt.colorbar(axes[1, 1].imshow(edge_detection_kernel), ax=axes[1, 1])

# Plot all outputs 2x2
fig2, axes2 = plt.subplots(2, 2, figsize=(10, 15))
axes2[0, 0].imshow(identity_output, cmap='gray')
axes2[0, 0].set_title('Identity')
axes2[0, 1].imshow(blur_output, cmap='gray')
axes2[0, 1].set_title('Blur')
axes2[1, 0].imshow(sharpen_output, cmap='gray')
axes2[1, 0].set_title('Sharpen')
axes2[1, 1].imshow(edge_detection_output, cmap='gray')
axes2[1, 1].set_title('Edge Detection')
fig2.suptitle('Outputs', fontsize=16)

plt.show()