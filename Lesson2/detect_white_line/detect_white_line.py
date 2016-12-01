import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read png image and convert to 0,255 bytescale
image = (mpimg.imread("test.png") * 255).astype('uint8')
print('This image is ', type(image), 'with dimensions', image.shape)

# Grab x and y size and make a copy of image
xsize = image.shape[1]
ysize = image.shape[0]
color_select = np.copy(image)
line_image = np.copy(image)

# Define a triangle region of interest
left_bottom = [0, 539]
right_bottom = [900, 539]
apex = [475, 320]

# Fit lines (y=Ax+B) to identify the 3 sided region of interest
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# Define color selection criteria
red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Do a bitwise or with "|" character to identify pixels below the threshold
color_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
				   (image[:,:,1] < rgb_threshold[1]) | \
				   (image[:,:,2] < rgb_threshold[2])

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX * fit_left[0] + fit_left[1])) & \
				    (YY > (XX * fit_right[0] + fit_right[1])) & \
				    (YY < (XX*fit_bottom[0] + fit_bottom[1]))

# Mask color selection
color_select[color_thresholds] = [0, 0, 0]

# Mask color and region selection
color_select[color_thresholds | ~region_thresholds] = [0, 0, 0]
# Color pixels red where both color and region selections met
line_image[~color_thresholds & region_thresholds] = [255, 0, 0]

# Display the image
plt.imshow(color_select)

# Save the image
mpimg.imsave("test-after.png", color_select)

