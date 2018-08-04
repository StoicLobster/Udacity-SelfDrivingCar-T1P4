## This script will define the functions used in the locate lane lines pipeline
## The end of this script will process a video file to locate and plot the lane lines

import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

## Unpickle Required Data
cam_mtx = pickle.load(open("camera_matrix.p","rb"))
dist_coef = pickle.load(open("camera_distortion_coefficients.p","rb"))
M = pickle.load(open("M.p","rb"))
Minv = pickle.load(open("Minv.p","rb"))

## Undistort Function
def undistort(img_RGB_in):
    # Input RGB distorted, Output RGB undistorted
    img_out = cv2.undistort(img_RGB_in, cam_mtx, dist_coef, None, cam_mtx)
    return(img_out)

# Sample undistort image
if (False):
    img = mpimg.imread('camera_cal/calibration1.jpg')
    dst_img = undistort(img)
    plt.figure(0)
    plt.imshow(img)
    plt.title('Original Image')
    plt.figure(1)
    plt.imshow(dst_img)
    plt.title('Undistorted Image')
    plt.show()
    
# Color Threshold Function
def color_thresh(img_RGB_in,RGB_out):
    # Input RGB undistorted, Output Binary (or RGB for video)
    # Convert image to HLS color space
    img_HLS = cv2.cvtColor(img_RGB_in, cv2.COLOR_RGB2HLS)
    # Extract S layer
    H_layer = img_HLS[:,:,0]
    L_layer = img_HLS[:,:,1]
    S_layer = img_HLS[:,:,2]
    # Apply threshold to S layer to identify white and yellow lane lines
    S_thresh = (90,255)
    L_thresh = 240 # Targeting white
    H_thresh = (20,35) # Targeting yellow
    img_out = np.zeros_like(H_layer)
    #bin_out[(S_layer >= S_thresh[0]) & (S_layer <= S_thresh[1])] = 1
    #bin_out[(S_layer >= S_thresh[0]) & (S_layer <= S_thresh[1]) \
    #        | (((H_layer >= H_thresh[0]) & (H_layer <= H_thresh[1])) \
    #        | (L_layer >= L_thresh))] = 1
    img_out[((H_layer >= H_thresh[0]) & (H_layer <= H_thresh[1])) \
            | (L_layer >= L_thresh)] = 1
            
    if (RGB_out):
        black_out_idxs = np.where(img_out == 0)
        img_out = np.copy(img_RGB_in)
        img_out[black_out_idxs[0],black_out_idxs[1],:] = 0
    
    return(img_out)

# Sample color threshold image
if (False):
    img = mpimg.imread('test_images/test6.jpg')
    thrsh_img = color_thresh(img,RGB_out=False)
    plt.figure(2)
    plt.imshow(thrsh_img, cmap='gray')
    plt.title('Color Threshold')
    plt.show()

## Perspective Transform to Top-Down View Function
def top_down_xfrm(img_RGB_in,frwd):
    # Input RGB undistorted, Output RGB top-down
    # frwd is bool that specifies if normal transform is requested (true) or inverse (false)
    img_size = (img_RGB_in.shape[1], img_RGB_in.shape[0])
    if (frwd):
        Xfrm = M
    else:
        Xfrm = Minv
    img_RGB_out = cv2.warpPerspective(img_RGB_in, Xfrm, img_size, flags=cv2.INTER_LINEAR)
    return(img_RGB_out)

# Sample top-down perspective transform on image
if (False):
    img = mpimg.imread('test_images/test6.jpg')
    warped = top_down_xfrm(img,frwd=True)
    plt.figure(3)
    plt.imshow(warped)
    plt.title('Top Down View Warp')
    plt.show()

## Gradient Threshold Function
def grad_thresh(img_RGB_in,RGB_out):
    # Input RGB top-down, Output Binary (or RGB for video)
    # RGB_out boolean can be used for video testing
    #Apply gradient threshold in x direction
    img_GRAY = cv2.cvtColor(img_RGB_in, cv2.COLOR_RGB2GRAY)
    grad_thresh = (20,100)
    abs_sobel = np.absolute(cv2.Sobel(img_GRAY, cv2.CV_64F, 1, 0))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    img_out = np.zeros_like(img_GRAY, dtype=np.uint8)
    img_out[(scaled_sobel >= grad_thresh[0]) & (scaled_sobel <= grad_thresh[1])] = 1
        
    if (RGB_out):
        black_out_idxs = np.where(img_out == 0)
        img_out = np.copy(img_RGB_in)
        img_out[black_out_idxs[0],black_out_idxs[1],:] = 0
        
    # print(out.shape)
    return(img_out)
    
# Sample gradient threshold image
if (False):
    img = mpimg.imread('test_images/test6.jpg')
    thrsh_img = grad_thresh(img,RGB_out=False)
    plt.figure(4)
    plt.imshow(thrsh_img, cmap='gray')
    plt.title('Gradient Threshold')
    plt.show()

## Calculate Histogram of Points

def hist(img_BIN_in):
    # Input Binary, Output Row Vector of Proportions
    # Normalize image (0-255, so normalize back to 0-1)
    # img_norm = img_BIN_in/255;
    #Grab only the bottom half of the image
    #Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img_BIN_in[img_BIN_in.shape[0]//2:,:]
    
    #Sum across image pixels vertically - make sure to set an `axis`
    #i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)
    
    return histogram

# Sample histogram
if (False):
    img = mpimg.imread('test_images/test6.jpg')
    histogram = hist(grad_thresh(top_down_xfrm(color_thresh(undistort(img),RGB_out=True),frwd=True),RGB_out=False))
    plt.figure(5)
    plt.plot(histogram)
    plt.title('Histogram')
    plt.show()

## Find Lane Pixels 

def find_lane_pixels(img_BIN_in):
    # Take a histogram of the bottom half of the image
    histogram = hist(img_BIN_in)
    # Create an output image to draw on and visualize the result
    img_RGB_out = np.dstack((img_BIN_in, img_BIN_in, img_BIN_in))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to re-center window
    minpix = 50
    
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(img_BIN_in.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img_BIN_in.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    #Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img_BIN_in.shape[0] - (window+1)*window_height
        win_y_high = img_BIN_in.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
    
        # Draw the windows on the visualization image
        cv2.rectangle(img_RGB_out,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(img_RGB_out,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window 
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
    
        # If you found > minpix pixels, re-center next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return leftx, lefty, rightx, righty, img_RGB_out
    
    
def fit_polynomial(img_BIN_in):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, img_RGB_out = find_lane_pixels(img_BIN_in)
    
    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_BIN_in.shape[0]-1, img_BIN_in.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    
    # Visualization 
    # Colors in the left and right lane regions
    img_RGB_out[lefty, leftx] = [255, 0, 0]
    img_RGB_out[righty, rightx] = [0, 0, 255]
    
    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    
    return img_RGB_out, left_fit, right_fit

# Sample polyfit
if (False):
    img = mpimg.imread('test_images/test6.jpg')
    img_RGB_out, left_fit, right_fit = fit_polynomial(grad_thresh(top_down_xfrm(color_thresh(undistort(img),RGB_out=True),frwd=True),RGB_out=False))
    plt.figure(6)
    plt.imshow(img_RGB_out)
    plt.title('Poly Fit')
    plt.show()
    
# Step through subsequent frames and search around polynomial for new ones
def search_around_poly(img_BIN_in, left_fit_prev, right_fit_prev):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100
    
    # Grab activated pixels
    nonzero = img_BIN_in.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # TO-DO: Set the area of search based on activated x-values 
    # within the +/- margin of our polynomial function 
    # Hint: consider the window areas for the similarly named variables 
    # in the previous quiz, but change the windows to our new search area 
    left_lane_inds = ((nonzerox > (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] - margin)) & (nonzerox < (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] - margin)) & (nonzerox < (right_fit_prev[0]*(nonzeroy**2) +  right_fit_prev[1]*nonzeroy + right_fit_prev[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit new polynomials
    img_RGB_out, left_fit, right_fit = fit_polynomial(img_BIN_in)
    
    # Visualization 
    # Create an image to draw on and an image to show the selection window
    #out_img = np.dstack((img_bin, img_bin, img_bin))*255
    window_img = np.zeros_like(img_RGB_out)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
    ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    img_RGB_out = cv2.addWeighted(img_RGB_out, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # End visualization steps 
    
    return img_RGB_out

## Process a given image (frame) to detect lane lines and return annotated image
def process_image(img_RGB_in):
    img_RGB_undistort = undistort(img_RGB_in)
    img_RGB_color = color_thresh(img_RGB_undistort,RGB_out=True)
    img_RGB_xfrm_frwd = top_down_xfrm(img_RGB_color,frwd=True)
    img_BIN_grad = grad_thresh(img_RGB_xfrm_frwd,RGB_out=False)
    img_RGB_out, left_fit, right_fit = fit_polynomial(img_BIN_grad);
    return(img_RGB_out)
    
## Process video
if (True):
    video_output  = 'output_videos/project_video_processed.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip = VideoFileClip("test_videos/project_video.mp4").subclip(0,5)
    video_clip = clip.fl_image(process_image) #NOTE: this function expects color images!!
    video_clip.write_videofile(video_output, audio=False)
