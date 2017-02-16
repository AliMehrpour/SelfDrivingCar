import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	
	if orient=='x':
		sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
	elif orient=='y':
		sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

	abs_sobel = np.absolute(sobel)

	scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

	grad_binary = np.zeros_like(scaled_sobel)
	grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

	return grad_binary


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

	mag = np.sqrt(sobelx**2 + sobely**2)

	scaled_sobel = np.uint8(255 * mag / np.max(mag))

	mag_binary = np.zeros_like(scaled_sobel)
	mag_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

	return mag_binary

def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

	abs_sobelx = np.absolute(sobelx)
	abs_sobely = np.absolute(sobely)

	direction = np.arctan2(abs_sobely, abs_sobelx)

	dir_binary = np.zeros_like(direction)
	dir_binary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1

	return dir_binary

ksize = 3
image = mpimg.imread('signs_vehicles_xygrad.png')

f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(50, 9))

ax1.imshow(image)
ax1.set_title('Orignial Image', fontsize=15)

gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
ax2.imshow(gradx, cmap='gray')
ax2.set_title('Gradient X', fontsize=15)

grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
ax3.imshow(grady, cmap='gray')
ax3.set_title('Gradient Y', fontsize=15)

mag = mag_thresh(image, sobel_kernel=ksize, thresh=(30, 100))
ax4.imshow(mag, cmap='gray')
ax4.set_title('Gradient', fontsize=15)

direction = dir_thresh(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
ax5.imshow(direction, cmap='gray')
ax5.set_title('Direction of the gradient', fontsize=15)

combined = np.zeros_like(direction)
combined[((gradx == 1) & (grady == 1)) | ((mag == 1) & (direction == 1))] = 1
ax6.imshow(combined, cmap='gray')
ax6.set_title('Combined', fontsize=15)

plt.show()