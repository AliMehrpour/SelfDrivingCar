"""
 Detect Lane Lines on Image and Video
 Steps:
 	1. Convert image to gray
 	2. Blur the image
 	3. Detect edges via Canny
 	4. Run Hough on edge detected image to find the lines
 	5. Draw the lines over the image
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Video processing
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# Tools
from tools import grayscale, gaussian_blur, canny, region_of_interest, hough_lines, weighted_image

def process_image(image):
	# Convert image to gray
	gray = grayscale(image)

	# Blur image
	gray_blur = gaussian_blur(gray, 3)

	# Detect edges via Canny
	low_threshold = 50
	high_threshold = 150
	edges = canny(gray_blur, low_threshold, high_threshold)

	# Specify region of interest
	xsize = image.shape[1]
	ysize = image.shape[0]
	xoffset = 40
	ymiddle = 320
	xcenter = xsize / 2

	vertices = np.array([[(xoffset, ysize), (xcenter - xoffset, ymiddle), (xsize - xoffset, ysize), (xcenter + xoffset, ymiddle)]], dtype = np.int32)
	masked_edges = region_of_interest(edges, vertices)

	# Run Hough on edge detected image
	rho = 2
	theta = np.pi / 180
	threshold = 15
	min_line_length = 40
	max_line_gap = 20
	hough_lines1 = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

	# Draw lines on the edge image
	color_edges = np.dstack((edges, edges, edges))
	weigthed_image = weighted_image(image, hough_lines1)

	return weigthed_image

"""
# Process Images
images = os.listdir("test_images/")
for image_name in images:
    image = mpimg.imread("test_images/" + image_name)
    print('This image is ', type(image), 'with dimensions:', image.shape)
    result_image = process_image(image)
    plt.imshow(result_image)
    plt.show()
"""

"""
Process one image
image = mpimg.imread("test.jpg")
result_image = process_image(image)
plt.imshow(result_image)
plt.show()
"""

# Process Video
input = VideoFileClip("challenge.mp4")
output = input.fl_image(process_image)
output.write_videofile('challenge_Output.mp4', audio=False)
