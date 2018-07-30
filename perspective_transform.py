## This script will calculate the perspective transform (and inverse) matrix
## and pickle them for later use

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

# Extract sample image
img = mpimg.imread('test_images/straight_lines1.jpg')

# Define source points
src_pts = np.float32(
    [[709,463], #Top Right
     [1034, 664], #Bottom Right
     [274, 664], #Bottom Left
     [573, 463]]) #Top Left

# Define destination points
img_size = (img.shape[1], img.shape[0])
print(img_size)
ofst = 0
dest_pts = np.float32(
    [[src_pts[1,0],ofst], #Top Right
     [src_pts[1,0], img_size[1] - ofst], #Bottom Right
     [src_pts[2,0], img_size[1] - ofst], #Bottom Left
     [src_pts[2,0], ofst]]) #Top Left

# Plot source image with selected source / destination points
plt.figure(0)
plt.imshow(img)
plt.plot(np.concatenate((src_pts[:,0],[src_pts[0,0]])),np.concatenate((src_pts[:,1],[src_pts[0,1]])),'r-')
plt.plot(src_pts[:,0],src_pts[:,1],'r.')
plt.plot(np.concatenate((dest_pts[:,0],[dest_pts[0,0]])),np.concatenate((dest_pts[:,1],[dest_pts[0,1]])),'b-')
plt.plot(dest_pts[:,0],dest_pts[:,1],'b.')
plt.title('Straight Road 1')

# Get transform matrices
M = cv2.getPerspectiveTransform(src_pts, dest_pts)
Minv = cv2.getPerspectiveTransform(dest_pts, src_pts)

# Pickle
pickle.dump(M,open("M.p","wb"))
pickle.dump(Minv,open("Minv.p","wb"))

# Test on sample image
warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
plt.figure(1)
plt.imshow(warped)
plt.title('Top Down View Warp')
plt.show()