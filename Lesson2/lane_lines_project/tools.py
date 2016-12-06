import cv2
import math
import numpy as np

def grayscale(img):
	"""
	Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
	return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def gaussian_blur(img, kernel_size):
	"""
	Applies a Gaussian Noise kernel
	"""
	return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshold, high_threshold):
	"""
	Applies the Canny transform
	"""
	return cv2.Canny(img, low_threshold, high_threshold)


def region_of_interest(img, vertices):
	"""
	Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """

    # Defining a blank mask to start with
	mask = np.zeros_like(img)

    # Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	if (len(img.shape) > 2):
		channel_count = img.shape[2]
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255

    # Filling pixels inside the polygon defined by "vertices" with the fill color    
	cv2.fillPoly(mask, vertices, ignore_mask_color)

    # Returning the image only where mask pixels are nonzero
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image


def draw_lines(img, lines, color = [255, 0, 0], thickness = 10):
	# Separating line segments by their slopes to decide which segments are part of
	# the left line vs. the right line. Then average the position of each the lines 
	# and extrapolate to the top na bottom of lane.
	left_segment_points = []
	right_segment_points = []
	min_slope_threshold = .5
	max_slope_threshold = 5
	for line in lines:
		for x1, y1, x2, y2 in line:
			# Calculate line slope. positive if line is going uphill from left to right and 
			# 						negative if line is going downhill from left to right 
			# And filter out noisy lines based on the slope
			slope = ((y2 - y1) / (x2 - x1))
			if (slope < max_slope_threshold and slope > min_slope_threshold): # Right lines points
				right_segment_points.append([x1,y1])
				right_segment_points.append([x2,y2])
			elif (slope > -max_slope_threshold and slope < -min_slope_threshold): # Left lines points
				left_segment_points.append([x1,y1])
				left_segment_points.append([x2,y2])

	top_y = 320 # Rawfully half of image
	bottom_y = img.shape[0] # Height of image

	# Now via fitLine() function, fitting a line to the points and calculate the its velocity, 
	# then calculate the x values via calculate_x() function
	# Draw right line
	if (len(right_segment_points) > 0):
		right_segment = np.array(right_segment_points)
		[r_vx, r_vy, r_cx, r_cy] = cv2.fitLine(right_segment, cv2.DIST_L2, 0, 0.01, 0.01)
		right_top_x = calculate_x(r_vx, r_vy, r_cx, r_cy, top_y)
		right_bottom_x = calculate_x(r_vx, r_vy, r_cx, r_cy, bottom_y)
		cv2.line(img, (right_top_x, top_y), (right_bottom_x, bottom_y), color, thickness)

	# Draw left line
	if (len(left_segment_points) > 0):
		left_segment = np.array(left_segment_points)
		[l_vx, l_vy, l_cx, l_cy] = cv2.fitLine(left_segment, cv2.DIST_L2, 0, 0.01, 0.01)
		left_top_x = calculate_x(l_vx, l_vy, l_cx, l_cy, top_y)
		left_bottom_x = calculate_x(l_vx, l_vy, l_cx, l_cy, bottom_y)
		cv2.line(img, (left_top_x, top_y), (left_bottom_x, bottom_y), color, thickness)

def calculate_x(vx, vy, x1, y1, y_ref):
    # Calculates 'x' matching 2 points on a line, its slope and a given 'y' coordinate.
    m = vy / vx
    b = y1 - ( m * x1 )
    x = ( y_ref - b ) / m
    return x

def hough_lines(img, rho, theta, threshold, main_line_len, max_line_gap):
	# Returns an image with hough lines drawn
	lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength = main_line_len, maxLineGap = max_line_gap)
	line_img = np.zeros((*img.shape, 3), dtype = np.uint8)
	draw_lines(line_img, lines)
	return line_img


def weighted_image(img, initial_img, α = 0.8, β = 1., λ = 0.):
	return cv2.addWeighted(initial_img, α, img, β, λ)