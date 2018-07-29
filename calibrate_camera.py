import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define base camera calibration file name
base_fname = 'camera_cal/calibration'

# Arrays to store calibration data
objPnts = [] #3D points of real world object
imgPnts = [] #2D points in image representation

# Prepare object points
nx = 9
ny = 6
objPt = np.zeros((ny*nx,3),np.float32)
objPt[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

# Loop over all calibration images
for i in range(20):
    fname = base_fname + str(i+1) + '.jpg'
    print(fname)
    img = mpimg.imread(fname)

    # Convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)
    
    # If corners found, add object and image points
    if ret == True:
        imgPnts.append(corners)
        objPnts.append(objPt)
        
        # Draw to display corners
        plt.figure(i)
        i = i + 1
        img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
        plt.imshow(img)
    
plt.show()
