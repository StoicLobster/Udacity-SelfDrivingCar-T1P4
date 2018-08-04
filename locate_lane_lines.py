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

# Sample histogram
if (False):
    img = mpimg.imread('test_images/test6.jpg')
    histogram = hist(grad_thresh(top_down_xfrm(color_thresh(undistort(img),RGB_out=True),frwd=True),RGB_out=False))
    plt.figure(5)
    plt.plot(histogram)
    plt.title('Histogram')
    plt.show()
    
    


# Sample polyfit
if (False):
    img = mpimg.imread('test_images/test6.jpg')
    img_BIN_in = grad_thresh(top_down_xfrm(color_thresh(undistort(img),RGB_out=True),frwd=True),RGB_out=False);
    leftx, lefty, rightx, righty, img_RGB_out = find_lane_pixels(img_BIN_in)
    img_RGB_out, left_fit, right_fit, left_fitx, right_fitx, ploty = fit_polynomial(img_BIN_in,img_RGB_out,leftx, lefty, rightx, righty)
    plt.figure(6)
    plt.imshow(img_RGB_out)
    plt.title('2nd Order Polynomial Fit')
    plt.show()
    
# Step through subsequent frames and search around polynomial for new ones
def search_around_poly(img_BIN_in,left_fit_prev,right_fit_prev):
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
    img_RGB_out = np.dstack((img_BIN_in, img_BIN_in, img_BIN_in))
    img_RGB_out, left_fit, right_fit, left_fitx, right_fitx, ploty = fit_polynomial(img_BIN_in,img_RGB_out,leftx,lefty,rightx,righty)
    
    # Visualization 
    # Create an image to draw on and an image to show the selection window
    # out_img = np.dstack((img_bin, img_bin, img_bin))*255
    window_img = np.zeros_like(img_RGB_out)
    # Color in left and right line pixels
    img_RGB_out[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    img_RGB_out[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
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
    
    return img_RGB_out, left_fit, right_fit, left_fitx, right_fitx, ploty

# Sample search around poly
if (True):
    img = mpimg.imread('test_images/test6.jpg')
    img_BIN_in = grad_thresh(top_down_xfrm(color_thresh(undistort(img),RGB_out=True),frwd=True),RGB_out=False);
    leftx, lefty, rightx, righty, img_RGB_out = find_lane_pixels(img_BIN_in)
    img_RGB_out, left_fit_prev, right_fit_prev, left_fitx, right_fitx, ploty = fit_polynomial(img_BIN_in,img_RGB_out,leftx, lefty, rightx, righty)
    img_RGB_out, left_fit, right_fit, left_fitx, right_fitx, ploty = search_around_poly(img_BIN_in,left_fit_prev,right_fit_prev)
    plt.figure(7)
    plt.imshow(img_RGB_out)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.title('Search Around Previous Polynomial')
    plt.show()

# Class to store and calculate both lane line parameters
class LaneLines():
    def __init__(self,frame_height,frame_width):
                
        # CONSTANTS
        # Frame height
        self.frame_height = frame_height
        # Frame width
        self.frame_width = frame_width
        self.midpoint = np.int(frame_height//2)
        # y values
        self.ploty = np.linspace(0, frame_height-1, frame_height)
        
        # HYPERPARAMETERS
        # Choose the number of sliding windows
        self.nwindows = 9
        # Set the width of the windows +/- margin
        self.margin_hist = 100
        # Set minimum number of pixels found to re-center window
        self.minpix = 50
        # Number of windows that must contain minpix number of pixels for lane line to be considered valid
        self.nwindow_fnd = 5
        # Set height of windows - based on nwindows above and image shape
        self.window_height = np.int(img_BIN_in.shape[0]//self.nwindows)
        # Define conversions in x and y from pixels space to meters
        self.x_width_pix = 700 #pixel width of lane
        self.y_height_pix = 720 #pixel height of lane (frame height)
        self.xm_per_pix = 3.7/self.x_width_pix # meters per pixel in x dimension
        self.ym_per_pix = 30/self.y_height_pix # meters per pixel in y dimension
        
        # FRAME
        # Binary image for current frame
        self.img_BIN_in = None
        # Histogram for current frame
        self.histogram = None
        # RGB image for output of current frame
        img_RGB_out = None
        
        # LINE PARAMETERS
        # was the left line detected in the current frame
        self.detected_L = False  
        # was the right line detected in the current frame
        self.detected_R = False 
        # x values of the last n fits of the left line
        self.recent_xfitted_L = [] 
        # x values of the last n fits of the right line
        self.recent_xfitted_R = [] 
        #average x values of the fitted left line over the last n iterations
        self.bestx_L = None    
        #average x values of the fitted right line over the last n iterations
        self.bestx_R = None   
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in [m]
        self.radius_of_curvature_L = None 
        self.radius_of_curvature_R = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        
    def hist(self):
        '''
        Calculate histogram of points
        '''
        #Grab only the bottom half of the image
        #Lane lines are likely to be mostly vertical nearest to the car
        #Sum across image pixels vertically - make sure to set an `axis`
        #i.e. the highest areas of vertical lines should be larger values
        self.histogram = np.sum(self.img_BIN_in[img_BIN_in.shape[0]//2:,:], axis=0)
        
        return
        
    def find_lane_pixels_hist(self):
        '''
        Find lane pixels with histogram method
        '''
        # Take a histogram of the bottom half of the image
        self.hist()
        # Create an output image to draw on and visualize the result
        self.img_RGB_out = np.dstack((self.img_BIN_in, self.img_BIN_in, self.img_BIN_in))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        leftx_base = np.argmax(self.histogram[:self.midpoint])
        rightx_base = np.argmax(self.histogram[self.midpoint:]) + self.midpoint
        
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
        
        # Counter of valid windows found
        cnt_wdw_fnd_L = 0
        cnt_wdw_fnd_R = 0        
        #Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img_BIN_in.shape[0] - (window+1)*self.window_height
            win_y_high = img_BIN_in.shape[0] - window*self.window_height
            win_xleft_low = leftx_current - self.margin_hist
            win_xleft_high = leftx_current + self.margin_hist
            win_xright_low = rightx_current - self.margin_hist
            win_xright_high = rightx_current + self.margin_hist
        
            # Draw the windows on the visualization image
            cv2.rectangle(self.img_RGB_out,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(self.img_RGB_out,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 
            
            # Identify the nonzero pixels in x and y within the window 
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
        
            # If you found > minpix pixels, re-center next window on their mean position (otherwise keep previous window x position)
            if len(good_left_inds) > self.minpix:
                cnt_wdw_fnd_L = cnt_wdw_fnd_L + 1
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix: 
                cnt_wdw_fnd_R = cnt_wdw_fnd_R + 1 
                self.detected_R = True      
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass
    
        # Determine if valid number of windows was found with pixels
        self.detected_L = cnt_wdw_fnd_L >= self.nwindow_fnd
        self.detected_R = cnt_wdw_fnd_R >= self.nwindow_fnd
        
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        return leftx, lefty, rightx, righty
    
    def fit_polynomial(self,x,y):
        # Fit a second order polynomial to data using `np.polyfit`
        coef_fit = np.polyfit(y, x, 2)
        
        # Generate x and y values for plotting
        fitx = coef_fit[0]*ploty**2 + coef_fit[1]*ploty + coef_fit[2]
        
        # Visualization 
        # Colors in the activated pixels
        img_RGB_out[y, x] = [255, 0, 0]
        # Colors in the poly line
        img_RGB_out[self.ploty, fitx] = [255, 255, 0]
        
        return coef_fit, fitx
    
    def find_lane_lines(self):
        '''
        Find lane lines with an appropriate method
        '''
        ## First find lane pixels
        # If both left and right detection from previous loop is false: Use histogram method  
        if (not(self.detected_L)) and (not(self.detected_R)):
            # Call histogram method to find pixel locations of lanes and determine current frame detection validity
            leftx, lefty, rightx, righty = self.find_lane_pixels_hist()
        else:
            # Call poly search method to find pixel locations of lanes and determine current frame detection validity
        
        # Next find lane polynomials
        # If only one lane detection in current frame is false: Use the valid lane to calculate polynomial and use for other side with pixel shift
            # Call polyfit for valid side and set other side detection to TRUE
        # Else:
            # Reuse best polyfit from last loop
            
        return
        
    def calc_rad_real(self):
        '''
        Calculates the radius of polynomial functions in meters.
        '''
        # Convert parabola coefficients into pixels
        A = self.xm_per_pix / (self.ym_per_pix**2) * self.best_fit[0]
        B = self.xm_per_pix / (self.ym_per_pix) * self.best_fit[1]
        
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = self.frame_height
        
        # Calculation of R_curve (radius of curvature)
        self.radius_of_curvature = ((1 + (2*A*y_eval + B)**2)**1.5) / np.absolute(2*A)
        
        return

    def draw_frame(self,warped):
        '''
        Draws the frame with desired polynomials in original image perspective
        '''
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        plt.imshow(result)

## Process a given image (frame) to detect lane lines and return annotated image
def process_image(img_RGB_in):
    img_RGB_undistort = undistort(img_RGB_in)
    img_RGB_color = color_thresh(img_RGB_undistort,RGB_out=True)
    img_RGB_xfrm_frwd = top_down_xfrm(img_RGB_color,frwd=True)
    img_BIN_grad = grad_thresh(img_RGB_xfrm_frwd,RGB_out=False)
    img_RGB_out, left_fit, right_fit = fit_polynomial(img_BIN_grad);
    return(img_RGB_out)
    
## Process video
if (False):
    video_output  = 'output_videos/project_video_processed.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip = VideoFileClip("test_videos/project_video.mp4").subclip(0,5)
    video_clip = clip.fl_image(process_image) #NOTE: this function expects color images!!
    video_clip.write_videofile(video_output, audio=False)
