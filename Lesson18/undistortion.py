import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

objp = np.zeros((6*8, 3), np.float32)
objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob('calibration_wide/GO*.jpg')

for idx, fname in enumerate(images):
	img = cv2.imread(fname)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
	if ret == True:
		objpoints.append(objp)
		imgpoints.append(corners)
"""
		cv2.drawChessboardCorners(img, (8, 6), corners, ret)
		cv2.imshow('img', img)
		cv2.waitKey(100)

cv2.destroyAllWindows()
"""

	
# Undistortion
import pickle

img = cv2.imread('calibration_wide/test_image.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration 
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('calibration_wide/test_undist.jpg', dst)

dist_pickle = {}
dist_pickle['mtx'] = mtx
dist_pickle['dist'] = dist
pickle.dump(dist_pickle, open('calibration_wide/wide_dist_pickle.p', 'wb'))

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(img)
ax1.set_title('Orignial Image')
ax2.imshow(dst)
ax2.set_title('Undistorted Image')

