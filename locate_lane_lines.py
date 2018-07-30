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
def undistort(img_dist):
    # Input RGB distorted, Output RGB undistorted
    dest = cv2.undistort(img_dist, cam_mtx, dist_coef, None, cam_mtx)
    return(dest)

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
def color_thresh(img_in,for_video):
    # Input RGB undistorted, Output Binary (or RGB for video)
    # Convert image to HLS color space
    hls = cv2.cvtColor(img_in, cv2.COLOR_RGB2HLS)
    # Extract S layer
    H_layer = hls[:,:,0]
    L_layer = hls[:,:,1]
    S_layer = hls[:,:,2]
    # Apply threshold to S layer to identify white and yellow lane lines
    S_thresh = (90,255)
    L_thresh = 240 # Targeting white
    H_thresh = (20,35) # Targeting yellow
    bin_out = np.zeros_like(H_layer)
    bin_out[(S_layer >= S_thresh[0]) & (S_layer <= S_thresh[1])] = 1
    #bin_out[(S_layer >= S_thresh[0]) & (S_layer <= S_thresh[1]) \
    #        | (((H_layer >= H_thresh[0]) & (H_layer <= H_thresh[1])) \
    #        | (L_layer >= L_thresh))] = 1
            
    if (for_video):
        print(bin_out.shape)
        black_out_idxs = np.where(bin_out == 0)
        bin_out = np.copy(img_in)
        bin_out[black_out_idxs[0],black_out_idxs[1],:] = 0
    
    return(bin_out)

# Sample color threshold image
if (False):
    img = mpimg.imread('test_images/test6.jpg')
    thrsh_img = color_thresh(img,for_video=False)
    plt.figure(2)
    plt.imshow(thrsh_img, cmap='gray')
    plt.title('Color Threshold')
    plt.show()

## Perspective Transform to Top-Down View Function
def top_down_xfrm(img_src,frwd):
    # Input RGB undistorted, Output RGB top-down
    # frwd is bool that specifies if normal transform is requested (true) or inverse (false)
    img_size = (img_src.shape[1], img_src.shape[0])
    if (frwd):
        Xfrm = M
    else:
        Xfrm = Minv
    img_dest = cv2.warpPerspective(img_src, Xfrm, img_size, flags=cv2.INTER_LINEAR)
    return(img_dest)

# Sample top-down perspective transform on image
if (False):
    img = mpimg.imread('test_images/straight_lines2.jpg')
    warped = top_down_xfrm(img,frwd=True)
    plt.figure(3)
    plt.imshow(warped)
    plt.title('Top Down View Warp')
    plt.show()

## Gradient Threshold Function
def grad_thresh(img_in,for_video):
    # Input RGB top-down, Output Binary (or RGB for video)
    # for_video boolean can be used for video testing
    #Apply gradient threshold in x direction
    gray = cv2.cvtColor(img_in, cv2.COLOR_RGB2GRAY)
    grad_thresh = (20,100)
    abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    bin_out = np.zeros_like(gray, dtype=np.uint8)
    bin_out[(scaled_sobel >= grad_thresh[0]) & (scaled_sobel <= grad_thresh[1])] = 1
        
    if (for_video):
        black_out_idxs = np.where(bin_out == 0)
        bin_out = np.copy(img_in)
        bin_out[black_out_idxs[0],black_out_idxs[1],:] = 0
        
    return(bin_out)
    
# Sample gradient threshold image
if (False):
    img = mpimg.imread('test_images/test6.jpg')
    thrsh_img = grad_thresh(img,for_video=False)
    plt.figure(4)
    plt.imshow(thrsh_img, cmap='gray')
    plt.title('Gradient Threshold')
    plt.show()


## Process a given image (frame) to detect lane lines and return annotated image
def process_image(img_in):
    img_undistort = undistort(img_in)
    img_color_thresh = color_thresh(img_undistort,for_video=True)
    img_top_down = top_down_xfrm(img_color_thresh,frwd=True)
    img_out = grad_thresh(img_top_down,for_video=True)
    return(img_out)
    
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
