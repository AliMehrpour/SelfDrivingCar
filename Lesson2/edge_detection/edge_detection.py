import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Read in the image and convert it to grayscale
image = (mpimg.imread('exit-ramp.png') * 255).astype('uint8')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
print('This image is ', type(image), 'with dimensions', image.shape)

# Define a kernel size for Gaussian smoothing / blurring
kernel_size = 5 # Must be an odd number (3, 5, 7, ...)
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

# Define parameters for Canny and apply
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Create a masked edges image using cv2.fillPoly() with three sides
mask = np.zeros_like(edges)
ignore_mask_color = 255

vertices = np.array([[(0, 539), (900, 539), (475, 280)]], dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)

#plt.imshow(masked_edges, cmap='Greys_r')
#plt.show()

# Define the Hough transform parameters
rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi / 180 # Angular resolution in radians of the Hough grid
threshold = 15 # minimum number of votes (intersections in Hough grid)
min_line_length = 40 # minimum number of pixels making up a line
max_line_gap = 20 # maximum gap in pixels between connectable line segments
line_image = np.copy(image) * 0 #creating a blank to draw lines on

# Run Hoough on edge detected image
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

# Iterate over the output lines and draw lines on the blank
for line in lines:
	for x1, y1, x2, y2 in line:
		cv2.line(line_image, (x1, y1), (x2,y2), (255, 0, 0), 10)

# Create a "color" binary image to combine with line image
color_edges = np.dstack((edges, edges, edges)) 

# Draw the lines on the edge image
lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
plt.imshow(lines_edges)
plt.show()