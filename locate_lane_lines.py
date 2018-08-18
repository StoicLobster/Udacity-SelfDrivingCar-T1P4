## This script will define the functions used in the locate lane lines pipeline
## The end of this script will process a video file to locate and plot the lane lines

import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import sys
from _operator import xor

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
    #img_out[(S_layer >= S_thresh[0]) & (S_layer <= S_thresh[1])] = 1
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
    plt.imshow(img)
    plt.title('Original Image')
    plt.figure(3)
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
    plt.figure(4)
    plt.imshow(img)
    plt.title('Original Image')
    plt.figure(5)
    plt.imshow(warped)
    plt.title('Top Down View Warp')
    plt.show()

## Gradient Threshold Function
def grad_thresh(img_RGB_in,RGB_out):
    # Input RGB top-down, Output Binary (or RGB for video)
    # RGB_out boolean can be used for video testing
    #Apply gradient threshold in x direction
    img_GRAY = cv2.cvtColor(img_RGB_in, cv2.COLOR_RGB2GRAY)
    grad_thresh = (10,100)
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
    plt.figure(6)
    plt.imshow(img)
    plt.title('Original Image')
    plt.figure(7)
    plt.imshow(thrsh_img, cmap='gray')
    plt.title('Gradient Threshold')
    plt.show()

# Class to store and calculate both lane line parameters
class LaneLines():
    def __init__(self,img_RGB_in,img_BIN_in):
        
        frame_height = img_RGB_in.shape[0]
        frame_width = img_RGB_in.shape[1]
                
        # CONSTANTS
        # Frame height
        self.frame_height = frame_height
        # Frame width
        self.frame_width = frame_width
        self.midpoint_width = np.int(frame_width//2)
        # y values
        self.ploty = np.linspace(0, frame_height-1, frame_height)
        # Polynomial fit dimension
        self.poly_fit_dim = 2
        
        # FRAME
        self.Frame = img_RGB_in
        # Binary image for current frame
        self.img_BIN_in = img_BIN_in
        # Histogram for current frame
        self.histogram = None
        # RGB image for output of current frame
        self.img_RGB_out = img_RGB_in
        # Current number of consecutive failed frames
        self.num_failed_frame_curr = 0
        # Number of frames processed
        self.frame_num = 0
        
        # TEXT
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.Ofst_Text_pos = (20,500)
        self.Rad_L_Text_pos = (20,550)
        self.Rad_R_Text_pos = (20,600)
        self.fontScale = 1
        self.fontColor = (255,255,255)
        self.lineType = 2
        
        # HYPERPARAMETERS
        # Choose the number of sliding windows
        self.nwindows = 9
        # Set the width of the windows +/- margin
        self.margin_hist = 100
        # Set the width of the windows +/- margin
        self.margin_poly = 100
        # Set minimum number of pixels found to re-center window
        self.minpix = 50
        # Number of windows that must contain minpix number of pixels for lane line to be considered valid
        self.nwindow_fnd = 5
        # Number of pixels that must be found for poly search method to be considered valid
        self.minpix_poly = 300
        # Set height of windows - based on nwindows above and image shape
        self.window_height = np.int(frame_height//self.nwindows)
        # Define conversions in x and y from pixels space to meters
        self.x_width_pix = 700 #pixel width of lane
        self.y_height_pix = 720 #pixel height of lane (frame height)
        self.xm_per_pix = 3.7/self.x_width_pix # meters per pixel in x dimension
        self.ym_per_pix = 30/self.y_height_pix # meters per pixel in y dimension
        # Number of frames that failed to find lane lines before reset
        self.num_failed_frame_alwd = 15
        # Number of frames for rolling average filter
        self.filt_size = 15
        
        # LINE PARAMETERS
        # was the left line detected in the current frame
        self.detected_L = False  
        self.detected_R = False 
        # x values of the last n fits of the left line
        self.x_fit_all_L = np.empty((0,self.ploty.size), dtype='float') 
        self.x_fit_all_R = np.empty((0,self.ploty.size), dtype='float')  
        #average x values of the fitted left line over the last n iterations
        self.x_fit_best_L = np.zeros((self.ploty.size), dtype='float')    
        self.x_fit_best_R = np.zeros((self.ploty.size), dtype='float')   
        #polynomial coefficients for the most recent fit
        self.coef_fit_current_L = np.zeros((self.poly_fit_dim+1), dtype='float')
        self.coef_fit_current_R = np.zeros((self.poly_fit_dim+1), dtype='float')  
        #polynomial coefficients for the previous n iterations
        self.coef_fit_all_L = np.empty((0,self.poly_fit_dim+1), dtype='float')  
        self.coef_fit_all_R = np.empty((0,self.poly_fit_dim+1), dtype='float')  
        #polynomial coefficients averaged over the last n iterations
        self.coef_fit_best_L = np.zeros((self.poly_fit_dim+1), dtype='float')  
        self.coef_fit_best_R = np.zeros((self.poly_fit_dim+1), dtype='float') 
        #radius of curvature of the line in [m]
        self.radius_of_curvature_L = 0 
        self.radius_of_curvature_R = 0 
        #distance in meters of vehicle center from the line
        self.center_line_offst = 0 
        #difference in fit coefficients between last and new fits
        # self.diffs = np.array([0,0,0], dtype='float') 
        
        return
    
    def update_frame(self,img_RGB_in):
        '''
        Stores the new frame in memory
        '''
        self.Frame = img_RGB_in
        self.histogram = None
        self.img_RGB_out = img_RGB_in
        
        return
        
    def hist(self):
        '''
        Calculate histogram of points
        '''
        #Grab only the bottom half of the image
        #Lane lines are likely to be mostly vertical nearest to the car
        #Sum across image pixels vertically - make sure to set an `axis`
        #i.e. the highest areas of vertical lines should be larger values
        self.histogram = np.sum(self.img_BIN_in[self.img_BIN_in.shape[0]//2:,:], axis=0)
        
        return
        
    def find_lane_pixels_hist(self):
        '''
        Find lane pixels with histogram method
        '''
        # Reset previous rolling average queues
        self.x_fit_all_L = np.empty((0,self.ploty.size), dtype='float') 
        self.x_fit_all_R = np.empty((0,self.ploty.size), dtype='float')
        self.coef_fit_all_L = np.empty((0,self.poly_fit_dim+1), dtype='float')  
        self.coef_fit_all_R = np.empty((0,self.poly_fit_dim+1), dtype='float') 
        # Take a histogram of the bottom half of the image
        self.hist()
        # Create an output image to draw on and visualize the result
        self.img_RGB_out = np.dstack((self.img_BIN_in, self.img_BIN_in, self.img_BIN_in))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint_height = np.int(self.histogram.shape[0]//2)
        leftx_base = np.argmax(self.histogram[:midpoint_height])
        rightx_base = np.argmax(self.histogram[midpoint_height:]) + midpoint_height
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = self.img_BIN_in.nonzero()
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
            win_y_low = self.img_BIN_in.shape[0] - (window+1)*self.window_height
            win_y_high = self.img_BIN_in.shape[0] - window*self.window_height
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
        
        # Create numpy arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    
        # Determine if valid number of windows was found with pixels
        self.detected_L = (self.frame_num == 0) or (cnt_wdw_fnd_L >= self.nwindow_fnd)
        self.detected_R = (self.frame_num == 0) or (cnt_wdw_fnd_R >= self.nwindow_fnd)
        
        # Color in left and right line pixels
        self.img_RGB_out[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        self.img_RGB_out[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        return leftx, lefty, rightx, righty
    
            
    def fit_polynomial(self,x,y):
        # Fit a second order polynomial to data using `np.polyfit`
        # coef_fit = [A, B, C] of y = A*x^2 + B*x + C
        coef_fit = np.polyfit(y, x, self.poly_fit_dim)
        
        # Generate x and y values for plotting
        x_fit = coef_fit[0]*self.ploty**2 + coef_fit[1]*self.ploty + coef_fit[2]
        
        # Limit x_fit by size of frame
        x_fit = np.minimum(np.maximum(x_fit,0),self.frame_width-1)
        
        # Visualization 
        # Colors in the activated pixels
        self.img_RGB_out[y, x] = [255, 0, 0]
        # Colors in the poly line
        self.img_RGB_out[self.ploty.astype(int), x_fit.astype(int)] = [255, 255, 0]
        
        return coef_fit, x_fit
    
    def find_lane_pixels_poly(self):
        '''
        Search around polynomial for new lane pixels
        '''        
        # Grab activated pixels
        nonzero = self.img_BIN_in.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Set the area of search based on activated x-values 
        # within the +/- margin of our polynomial function (from previous frame)
        left_lane_inds = ((nonzerox > (self.coef_fit_current_L[0]*(nonzeroy**2) + self.coef_fit_current_L[1]*nonzeroy + self.coef_fit_current_L[2] - self.margin_poly)) & (nonzerox < (self.coef_fit_current_L[0]*(nonzeroy**2) + self.coef_fit_current_L[1]*nonzeroy + self.coef_fit_current_L[2] + self.margin_poly)))
        right_lane_inds = ((nonzerox > (self.coef_fit_current_R[0]*(nonzeroy**2) + self.coef_fit_current_R[1]*nonzeroy + self.coef_fit_current_R[2] - self.margin_poly)) & (nonzerox < (self.coef_fit_current_R[0]*(nonzeroy**2) +  self.coef_fit_current_R[1]*nonzeroy + self.coef_fit_current_R[2] + self.margin_poly)))
        
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        # Determine pixel find validity
        self.detected_L = len(leftx) > self.minpix_poly
        self.detected_R = len(rightx) > self.minpix_poly
        
        if (self.detected_L and self.detected_R):
            # Prepare output RGB image
            self.img_RGB_out = np.dstack((self.img_BIN_in, self.img_BIN_in, self.img_BIN_in))
            
            # Visualization 
            # Create an image to draw on and an image to show the selection window
            # out_img = np.dstack((img_bin, img_bin, img_bin))*255
            window_img = np.zeros_like(self.img_RGB_out)
            # Color in left and right line pixels
            self.img_RGB_out[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            self.img_RGB_out[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            
            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            coef_tmp_L, x_fit_L = self.fit_polynomial(leftx,lefty)
            coef_tmp_R, x_fit_R = self.fit_polynomial(rightx,righty)
            left_line_window1 = np.array([np.transpose(np.vstack([x_fit_L-self.margin_poly, self.ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([x_fit_L+self.margin_poly, self.ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([x_fit_R-self.margin_poly, self.ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([x_fit_R+self.margin_poly, self.ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))
            
            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
            self.img_RGB_out = cv2.addWeighted(self.img_RGB_out, 1, window_img, 0.3, 0)
            
            # Plot the polynomial lines onto the image
            # plt.plot(left_fitx, ploty, color='yellow')
            # plt.plot(right_fitx, ploty, color='yellow')
            # End visualization steps 
        
        return leftx, lefty, rightx, righty
    
    def calc_best(self):
        '''
        Perform rolling average on polynomials to determine best fit.
        '''
        # Reset best
        self.coef_fit_best_L = np.zeros((self.poly_fit_dim+1), dtype='float')  
        self.coef_fit_best_R = np.zeros((self.poly_fit_dim+1), dtype='float') 
        self.x_fit_best_L = np.zeros((self.ploty.size), dtype='float')    
        self.x_fit_best_R = np.zeros((self.ploty.size), dtype='float')  
        
        # Check if size of queue is larger than filter size
        if (self.x_fit_all_L.shape[0] > self.filt_size):
            self.x_fit_all_L = np.delete(self.x_fit_all_L,(0),axis=0)
            self.x_fit_all_R = np.delete(self.x_fit_all_R,(0),axis=0)
            self.coef_fit_all_L = np.delete(self.coef_fit_all_L,(0),axis=0)
            self.coef_fit_all_R = np.delete(self.coef_fit_all_R,(0),axis=0)
        
        # Loop through and compute average
        n = self.x_fit_all_L.shape[0]
        for row in range(n):
            for col_x_fit in range(self.x_fit_all_L.shape[1]):
                self.x_fit_best_L[col_x_fit] = self.x_fit_best_L[col_x_fit] + self.x_fit_all_L[row,col_x_fit]
                self.x_fit_best_R[col_x_fit] = self.x_fit_best_R[col_x_fit] + self.x_fit_all_R[row,col_x_fit]
            for col_coef_fit in range(self.coef_fit_all_L.shape[1]):
                self.coef_fit_best_L[col_coef_fit] = self.coef_fit_best_L[col_coef_fit] + self.coef_fit_all_L[row,col_coef_fit]
                self.coef_fit_best_R[col_coef_fit] = self.coef_fit_best_R[col_coef_fit] + self.coef_fit_all_R[row,col_coef_fit]   
                
        self.x_fit_best_L = self.x_fit_best_L/n    
        self.x_fit_best_R = self.x_fit_best_R/n
        self.coef_fit_best_L = self.coef_fit_best_L/n
        self.coef_fit_best_R = self.coef_fit_best_R/n
        
        return
    
    def calc_rad_real(self):
        '''
        Calculates the radius of polynomial functions in meters.
        '''
        # Convert parabola coefficients into pixels
        A_L = self.xm_per_pix / (self.ym_per_pix**2) * self.coef_fit_best_L[0]
        B_L = self.xm_per_pix / (self.ym_per_pix) * self.coef_fit_best_L[1]
        A_R = self.xm_per_pix / (self.ym_per_pix**2) * self.coef_fit_best_R[0]
        B_R = self.xm_per_pix / (self.ym_per_pix) * self.coef_fit_best_R[1]
        
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = self.frame_height - 1
        
        # Calculation of R_curve (radius of curvature)
        self.radius_of_curvature_L = ((1 + (2*A_L*y_eval + B_L)**2)**1.5) / np.absolute(2*A_L)
        self.radius_of_curvature_R = ((1 + (2*A_R*y_eval + B_R)**2)**1.5) / np.absolute(2*A_R)
        
        return
    
    def calc_offset(self):
        '''
        Calculates the offset between vehicle and center of lane
        '''
        self.center_line_offst = abs(self.midpoint_width - (self.x_fit_best_L[-1] + self.x_fit_best_R[-1])/2) * self.xm_per_pix
        
        return
    
    def find_lane_lines(self):
        '''
        Find lane lines with an appropriate method
        '''
        ## Find lane pixels
        # If left and right detection from previous loop is false: Use histogram method  
        if (not(self.detected_L)) and (not(self.detected_R)):
            print("Histogram search method used.")
            # Call histogram method to find pixel locations of lanes and determine current frame detection validity
            leftx, lefty, rightx, righty = self.find_lane_pixels_hist()
        else:
            print("Polynomial search method used")
            # Call poly search method to find pixel locations of lanes and determine current frame detection validity
            leftx, lefty, rightx, righty = self.find_lane_pixels_poly()
            if (not(self.detected_L)) or (not(self.detected_R)):
                print("Polynomial search method failed. Histogram search method used.") 
                # Neither lane was found, must use histogram method
                leftx, lefty, rightx, righty = self.find_lane_pixels_hist()    
                
        ## Check if both lane lines were found
        if (self.detected_L and self.detected_R):
            # Reset failed counter            
            self.num_failed_frame_curr = 0
            
            # Fit new polynomials for both lanes       
            self.coef_fit_current_L, x_fit_L = self.fit_polynomial(leftx,lefty)
            self.coef_fit_current_R, x_fit_R = self.fit_polynomial(rightx,righty)
            
            # Append x_fit to list
            self.x_fit_all_L = np.vstack((self.x_fit_all_L, x_fit_L))
            self.x_fit_all_R = np.vstack((self.x_fit_all_R, x_fit_R))
            
            # Append coefficients to list
            self.coef_fit_all_L = np.vstack((self.coef_fit_all_L, self.coef_fit_current_L))
            self.coef_fit_all_R = np.vstack((self.coef_fit_all_R, self.coef_fit_current_R))
            
            # Calculate rolling average
            self.calc_best()
        else:
            # Increment failed counter            
            self.num_failed_frame_curr = self.num_failed_frame_curr + 1
            print("Number of failed frames: " + str(self.num_failed_frame_curr))
            # Do not compute new polynomial, use previous best
            # Check if number of consecutive failed frames has exceed max
            if (self.num_failed_frame_curr > self.num_failed_frame_alwd):
                print("Number of consecutive failed frames exceeded.")
                sys.exit()
        
        # Calculate radius of curvature
        self.calc_rad_real()
        
        # Calculate center line offset
        self.calc_offset()
        
#         elif (self.detected_L and not(self.detected_R)):
#             print("Left detected and right not.")
#             # Shift left lane line to right side
#             
#         elif (not(self.detected_L) and self.detected_R):
#             print("Right detected and left not.")
#             # Shift right lane line to left side
            
        return
        
    def draw_frame(self,img_RGB_in):
        '''
        Draws the frame with desired polynomials in original image perspective
        '''
        print("\n")
        #print("Processing Frame # " + str(self.frame_num))
        # Store new frame
        self.update_frame(img_RGB_in)
        
        # Calculate binary image of color and gradient thresholds
        self.img_BIN_in = grad_thresh(top_down_xfrm(color_thresh(undistort(img_RGB_in),RGB_out=True),frwd=True),RGB_out=False)
        
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(self.img_BIN_in).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Find lane lines
        self.find_lane_lines()
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.x_fit_best_L, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.x_fit_best_R, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img_RGB_in.shape[1], img_RGB_in.shape[0])) 
        # Combine the result with the original image
        self.Frame = cv2.addWeighted(img_RGB_in, 1, newwarp, 0.3, 0)

        # Draw text on image   
        cv2.putText(self.Frame,"Lane Center Offset [m]: " + str(round(self.center_line_offst,2)),self.Ofst_Text_pos,self.font,self.fontScale,self.fontColor,self.lineType)   
        cv2.putText(self.Frame,"Radius Left [m]: " + str(round(self.radius_of_curvature_L,0)),self.Rad_L_Text_pos,self.font,self.fontScale,self.fontColor,self.lineType)
        cv2.putText(self.Frame,"Radius Right [m]: " + str(round(self.radius_of_curvature_R,0)),self.Rad_R_Text_pos,self.font,self.fontScale,self.fontColor,self.lineType)
        
        self.frame_num = self.frame_num + 1
        #print("Left Radius: " + str(self.radius_of_curvature_L))
        #print("Right Radius: " + str(self.radius_of_curvature_R))
        #print("Lane Center Offset: " + str(lane_lines.center_line_offst))
        
        #return(self.Frame)
        return(self.img_RGB_out)
                
# Sample histogram
if (False):
    img = mpimg.imread('test_images/test6.jpg')
    img_BIN_in = grad_thresh(top_down_xfrm(color_thresh(undistort(img),RGB_out=True),frwd=True),RGB_out=False);
    lane_lines = LaneLines(img,img_BIN_in)
    lane_lines.hist()
    histogram = lane_lines.histogram
    plt.figure(8)
    plt.imshow(img)
    plt.title('Original Image')
    plt.figure(9)
    plt.plot(histogram)
    plt.title('Histogram')
    plt.show()
    
# Sample polyfit with histogram search
if (False):
    img = mpimg.imread('test_images/test6.jpg')
    plt.figure(10)
    plt.imshow(img)
    plt.title('Original Image')
    img_BIN_in = grad_thresh(top_down_xfrm(color_thresh(undistort(img),RGB_out=True),frwd=True),RGB_out=False)
    lane_lines = LaneLines(img,img_BIN_in)
    # Search for lane lines using histogram method
    leftx, lefty, rightx, righty = lane_lines.find_lane_pixels_hist()
    # Fit new polynomials for both lanes 
    lane_lines.coef_fit_current_L, x_fit_L = lane_lines.fit_polynomial(leftx,lefty)
    lane_lines.coef_fit_current_R, x_fit_R = lane_lines.fit_polynomial(rightx,righty)
    print("Current Left Coefficients: " + str(lane_lines.coef_fit_current_L))
    print("Current Right Coefficients: " + str(lane_lines.coef_fit_current_R))
    plt.figure(11)
    plt.imshow(lane_lines.img_RGB_out)
    plt.title('2nd Order Polynomial Fit')
    # Sample search around poly
    if (False):
        # Append x_fit to list
        lane_lines.x_fit_all_L = np.vstack((lane_lines.x_fit_all_L, x_fit_L))
        lane_lines.x_fit_all_R = np.vstack((lane_lines.x_fit_all_R, x_fit_R))
        # Append coefficients to list
        lane_lines.coef_fit_all_L = np.vstack((lane_lines.coef_fit_all_L, lane_lines.coef_fit_current_L))
        lane_lines.coef_fit_all_R = np.vstack((lane_lines.coef_fit_all_R, lane_lines.coef_fit_current_R))
        print("All Left Coefficients: " + str(lane_lines.coef_fit_all_L))
        print("All Right Coefficients: " + str(lane_lines.coef_fit_all_R))
        # Calculate rolling average
        lane_lines.calc_best()
        print("Best Left Coefficients: " + str(lane_lines.coef_fit_best_L))
        print("Best Right Coefficients: " + str(lane_lines.coef_fit_best_R))
        # Calculate real radius of curvature
        lane_lines.calc_rad_real()
        print("Left Radius: " + str(lane_lines.radius_of_curvature_L))
        print("Right Radius: " + str(lane_lines.radius_of_curvature_R))
        lane_lines.calc_offset()
        print("Center Lane Offset: " + str(lane_lines.center_line_offst))
        # Search for lane lines around previous best polynomial
        leftx, lefty, rightx, righty = lane_lines.find_lane_pixels_poly()
        # Fit new polynomials for both lanes       
        lane_lines.coef_fit_current_L, x_fit_L = lane_lines.fit_polynomial(leftx,lefty)
        lane_lines.coef_fit_current_R, x_fit_R = lane_lines.fit_polynomial(rightx,righty)
        plt.figure(12)
        plt.imshow(lane_lines.img_RGB_out)
        plt.title('Search Around Previous Polynomial')
    plt.show()
    
# Test full pipeline
if (False):
    img = mpimg.imread('test_images/test6.jpg')
    lane_lines = LaneLines(img,img)
    plt.figure(13)
    plt.imshow(img)
    plt.title('Original Image')
    plt.figure(14)
    plt.imshow(lane_lines.draw_frame(img))
    plt.title('Found Lines')
    plt.show()
    
## Process video
if (True):
    img = mpimg.imread('test_images/test6.jpg')
    lane_lines = LaneLines(img,img)
    video_output  = 'output_videos/project_video_processed.mp4'
    #video_output  = 'output_videos/project_video_processed.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip = VideoFileClip("test_videos/project_video.mp4").subclip(23,28)
    video_clip = clip.fl_image(lane_lines.draw_frame) #NOTE: this function expects color images!!
    video_clip.write_videofile(video_output, audio=False)
